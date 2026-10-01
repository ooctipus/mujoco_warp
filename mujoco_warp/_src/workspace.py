# Copyright 2025 The Newton Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Prepared native step scratch with explicit allocation and lifetime ownership."""

import dataclasses
import math

import warp as wp
from gpu_components import fields as field_ops
from gpu_components import graph as graph_ops
from gpu_components.field_data import FieldStorage
from gpu_components.graph_data import GraphKernelBinding
from gpu_components.graph_data import GraphUpdateTable
from gpu_components.graph_data import KernelParameterBinding

from mujoco_warp._src import solver
from mujoco_warp._src import types
from mujoco_warp._src.collision_core import CollisionContext
from mujoco_warp._src.collision_driver import MJ_COLLISION_TABLE
from mujoco_warp._src.collision_driver import CollisionType
from mujoco_warp._src.math import upper_trid_index


@dataclasses.dataclass(frozen=True)
class WorkspaceFieldSpec:
  """Scratch requirement; capacity_domain declares its independent capacity axis."""

  name: str
  shape: tuple[int, ...]
  dtype: type
  capacity_domain: str


@dataclasses.dataclass(eq=False)
class StepBindings:
  """Borrowed native capacity domains and their single captured binding ledger.

  Keep all three storage owners alive through graph retirement. World launches use
  world_storage.protected_count; candidate and CCD launches use their ready_count.
  Admission must keep those counts within physically ready rows. Set updates before
  the first recorded operation. Only native binding operations mutate the ledgers
  and recording snapshot; do not replace their storage or count sources. A composing
  engine also sets recording_failed on application or capture failure. Never clear
  this latch or publish a graph after it becomes true.
  """

  world_storage: FieldStorage
  contact_storage: FieldStorage
  ccd_storage: FieldStorage
  updates: GraphUpdateTable | None = None
  recording_failed: bool = False
  bindings: list[GraphKernelBinding] = dataclasses.field(default_factory=list)
  operations: list[dict] = dataclasses.field(default_factory=list)
  _recording_binding: tuple | None = None


def _binding_layout(bindings):
  if bindings is None:
    return None
  return tuple(
    (id(storage), storage.capacity, id(storage.device), id(count), _StepWorkspace._layout(count))
    for storage, count in (
      (bindings.world_storage, bindings.world_storage.protected_count),
      (bindings.contact_storage, bindings.contact_storage.ready_count),
      (bindings.ccd_storage, bindings.ccd_storage.ready_count),
    )
  ) + (id(bindings.bindings), id(bindings.operations))


def _validate_bindings(bindings):
  if not isinstance(bindings, StepBindings):
    raise TypeError("Prepared bindings must be a native StepBindings record")
  if bindings.recording_failed:
    raise RuntimeError("The native step program has a failed recording")
  if not all(
    isinstance(storage, FieldStorage) for storage in (bindings.world_storage, bindings.contact_storage, bindings.ccd_storage)
  ):
    raise TypeError("Native capacity domains must borrow FieldStorage records")
  device = bindings.world_storage.device
  for storage, count in (
    (bindings.world_storage, bindings.world_storage.protected_count),
    (bindings.contact_storage, bindings.contact_storage.ready_count),
    (bindings.ccd_storage, bindings.ccd_storage.ready_count),
  ):
    if storage.closed or storage.service_failed:
      raise RuntimeError("Native step storage is closed or quarantined")
    if type(storage.capacity) is not int or not 0 < storage.capacity <= 2**31 - 1:
      raise ValueError("Native storage capacity must be a positive int32 count")
    if (
      storage.device != device
      or not isinstance(count, wp.array)
      or count.device != device
      or count.dtype != wp.int32
      or count.shape != (1,)
      or not count.is_contiguous
    ):
      raise ValueError("Native counts must be contiguous int32 scalars on the storage device")
  updates = bindings.updates
  if updates is not None and (
    not isinstance(updates, GraphUpdateTable)
    or updates.device != device
    or updates.enable_count is not bindings.world_storage.protected_count
    or updates.enable_count_maximum != bindings.world_storage.capacity
  ):
    raise ValueError("Native updates must use the exact world protected count and capacity")
  if bindings._recording_binding is not None and bindings._recording_binding != (_binding_layout(bindings), id(updates)):
    raise ValueError("Native recording binding changed; restore its original storage, counts, updates and ledgers")


def validate_step_launch(bindings, kernel, extent_domain, extent_axis, parameter_domains):
  """Resolve native count declarations before emitting a launch; mutate no recording state.

  A dynamic extent is the leading axis of world, candidate or CCD storage. Fixed
  worker grids explicitly use extent_domain=None and extent_axis=None. Named int32
  scalar arguments may independently use any of those domains. Returned descriptors
  borrow their exact count arrays; parameter indices include Warp's launch bounds.
  """
  _validate_bindings(bindings)
  owners = {"world": bindings.world_storage, "candidate": bindings.contact_storage, "ccd": bindings.ccd_storage}
  if extent_domain is not None and extent_domain not in owners:
    raise ValueError(f"Unknown native extent domain: {extent_domain!r}")
  if extent_domain is None and extent_axis is not None:
    raise ValueError("A fixed worker grid must explicitly omit its extent axis")
  if extent_domain is not None and (type(extent_axis) is not int or extent_axis != 0):
    raise ValueError("Native dynamic extents require explicit leading axis zero")
  if parameter_domains is not None and len(parameter_domains) > 4:
    raise ValueError("A native launch supports at most four count parameters")
  sources = {
    name: (owner.protected_count if name == "world" else owner.ready_count, owner.capacity) for name, owner in owners.items()
  }
  labels = {argument.label: (index + 1, argument.type) for index, argument in enumerate(kernel.adj.args)}
  parameters = []
  for name, domain in (parameter_domains or {}).items():
    if name not in labels or domain not in sources:
      raise ValueError(f"Unknown native count argument or domain: {name!r}, {domain!r}")
    index, dtype = labels[name]
    if dtype not in (int, wp.int32):
      raise ValueError(f"Native count argument must be int32: {name!r}")
    parameters.append(KernelParameterBinding(index, *sources[domain]))
  return sources[extent_domain][0] if extent_domain is not None else None, tuple(parameters)


def _begin_recording(bindings):
  if bindings.updates is None:
    raise RuntimeError("Prepare native graph updates before recording step operations")
  device = bindings.world_storage.device
  graph = device.captures.get(wp.get_stream(device))
  if graph is None:
    raise RuntimeError("Record native step operations inside a Warp-managed graph capture")
  if getattr(graph, "_preparation_failed", False):
    raise RuntimeError("Discard this graph after failed preparation")
  if bindings._recording_binding is None:
    bindings._recording_binding = _binding_layout(bindings), id(bindings.updates)
  owners = getattr(graph, "_resource_owners", ())
  if not any(owner is bindings for owner in owners):
    graph._resource_owners = (*owners, bindings)
    for storage in (bindings.world_storage, bindings.contact_storage, bindings.ccd_storage):
      field_ops.retain_graph(storage, graph)


def _fail_recording(bindings):
  if not isinstance(bindings, StepBindings):
    return
  bindings.recording_failed = True
  try:
    device = (
      bindings.updates.device
      if isinstance(bindings.updates, GraphUpdateTable)
      else bindings.world_storage.protected_count.device
    )
    graph = device.captures.get(wp.get_stream(device) if device.is_cuda else None)
    if graph is not None:
      graph._preparation_failed = True
  except BaseException:
    # A malformed borrowed descriptor must not mask the error that poisoned it.
    # The composition root must also reject the native recording_failed latch.
    pass


def bind_step_launch(bindings, kernel, dim, extent_domain, *, extent_axis=0, parameter_domains=None):
  """Bind the just-emitted native kernel and retain its explicit count sources.

  Call validate_step_launch before emission when declarations come from callers.
  Zero-sized operations claim no node. Any other failure poisons this binding
  record and the active graph, because a launch may already have been emitted.
  """
  try:
    extent, parameters = validate_step_launch(bindings, kernel, extent_domain, extent_axis, parameter_domains)
    dimensions = (dim,) if isinstance(dim, int) else tuple(dim)
    if not all(dimensions):
      return
    _begin_recording(bindings)
    binding = GraphKernelBinding(
      graph_ops.register_last_kernel_node(bindings.updates),
      launch_rank=kernel.adj.kernel_dim,
      extent_axis=extent_axis,
      extent_source=extent,
      parameters=parameters,
    )
    bindings.bindings.append(binding)
    bindings.operations.append(
      {
        "operation": "launch",
        "kernel": kernel.key,
        "module": kernel.func.__module__,
        "function": kernel.func.__qualname__,
        "dim": list(dimensions),
        "extent_domain": extent_domain,
        "extent_axis": extent_axis,
        "parameter_domains": dict(parameter_domains or {}),
        "launch_rank": binding.launch_rank,
        "node": binding.node,
      }
    )
  except BaseException:
    _fail_recording(bindings)
    raise


def _fill_step_rows(bindings, array, value, domain):
  try:
    _validate_bindings(bindings)
    _begin_recording(bindings)
    owners = {"world": bindings.world_storage, "candidate": bindings.contact_storage, "ccd": bindings.ccd_storage}
    if domain not in owners:
      raise ValueError(f"Unknown native memory domain: {domain!r}")
    owner = owners[domain]
    field_ops.fill(owner, array, value, count=owner.protected_count if domain == "world" else owner.ready_count)
    bindings.operations.append({"operation": "fill", "domain": domain, "field": field_ops.lookup(owner, array).name})
  except BaseException:
    _fail_recording(bindings)
    raise


def _copy_step_rows(bindings, destination, source, domain):
  try:
    _validate_bindings(bindings)
    _begin_recording(bindings)
    owners = {"world": bindings.world_storage, "candidate": bindings.contact_storage, "ccd": bindings.ccd_storage}
    if domain not in owners:
      raise ValueError(f"Unknown native memory domain: {domain!r}")
    owner = owners[domain]
    field_ops.copy(owner, destination, source, count=owner.protected_count if domain == "world" else owner.ready_count)
    field = field_ops.lookup(owner, destination) or field_ops.lookup(owner, source)
    bindings.operations.append({"operation": "copy", "domain": domain, "field": field.name})
  except BaseException:
    _fail_recording(bindings)
    raise


class _StepWorkspace:
  """Internal owner of prepared scratch and borrowed native stage views.

  Prepare with make_step_workspace. One workspace may serve ordered substeps and
  both collision passes; concurrent steps require separate workspaces. Model and
  Data storage must remain unchanged. Captured steps retain this owner on the graph.
  Compact solver and collision views are private engine implementation; callers
  provide only the original Model/Data and declared scratch descriptors.
  """

  def __init__(self, model, data, specs, solver_model, solver_data, arrays=None, bindings=None):
    self.model, self.data = model, data
    self.world_live_count = None if bindings is None else bindings.world_storage.protected_count
    self.bindings = bindings
    self._execution_binding = self.world_live_count, self._layout(self.world_live_count), bindings
    self._binding_layout = _binding_layout(bindings)
    self.device = data.qpos.device
    self._solver_model, self._solver_data = solver_model, solver_data
    self.arrays, self._ledger = {}, []
    offset = 0
    for spec in specs:
      name, shape, dtype, capacity_domain = spec.name, spec.shape, spec.dtype, spec.capacity_domain
      nbytes = math.prod(shape) * wp.types.type_size_in_bytes(dtype)
      offset = (offset + 127) // 128 * 128
      field = dict(
        name=name,
        shape=shape,
        dtype=dtype,
        allocation_offset_bytes=offset,
        payload_bytes=nbytes,
        capacity_domain=capacity_domain,
      )
      field["world_axis"] = 0 if capacity_domain == "world" else None
      self._ledger.append(field)
      offset += nbytes
    self.storage = None
    if arrays is None:
      self.storage = wp.empty(offset, dtype=wp.uint8, device=self.device)
      for field in self._ledger:
        array = (
          wp.array(
            ptr=self.storage.ptr + field["allocation_offset_bytes"],
            shape=field["shape"],
            dtype=field["dtype"],
            device=self.device,
          )
          if field["payload_bytes"]
          else wp.empty(field["shape"], dtype=field["dtype"], device=self.device)
        )
        array.workspace_storage = self.storage
        self.arrays[field["name"]] = array
    else:
      if set(arrays) != {spec.name for spec in specs}:
        raise ValueError("Caller-owned scratch must supply exactly the declared field names")
      for spec in specs:
        array = arrays[spec.name]
        expected = spec.shape, spec.dtype, self.device
        if not isinstance(array, wp.array) or (array.shape, array.dtype, array.device) != expected:
          raise ValueError(f"Caller-owned scratch has incompatible descriptor: {spec.name}")
        width = wp.types.type_size_in_bytes(array.dtype)
        for axis in range(array.ndim - 1, 0, -1):
          if array.size and array.strides[axis] != width:
            raise ValueError(f"Scratch inner dimensions must be contiguous: {spec.name}")
          width *= array.shape[axis]
        alignment = wp.types.type_size_in_bytes(getattr(array.dtype, "_wp_scalar_type_", array.dtype))
        if array.size and (not array.ptr or array.ptr % alignment or array.strides[0] < width or array.strides[0] % alignment):
          raise ValueError(f"Scratch rows must have a nonoverlapping aligned stride: {spec.name}")
        if spec.name in ("solver.h", "solver.hfactor") and array.size and solver_model.nv > solver._BLOCK_CHOLESKY_DIM:
          # Blocked Cholesky explicitly opts its matrix tiles into aligned=True.
          if array.ptr % 16 or array.strides[0] % 16 or array.strides[1] % 16:
            raise ValueError(f"Blocked Cholesky matrix needs 16-byte base and row strides: {spec.name}")
        self.arrays[spec.name] = array
        self._ledger[len(self.arrays) - 1]["allocation_offset_bytes"] = None
    self._collision = CollisionContext(
      **{name: self.arrays[name] for name in ("collision_pair", "collision_pairid", "collision_worldid")}
    )
    self._solver_context = types.SolverContext(
      **{name: self.arrays["solver." + name] for name, _, _, _ in solver._solver_context_layout(solver_model, solver_data)}
    )
    self._solver_context.compact_m_full = model
    self._solver_context.compact_d_full = data
    self._data_layout = self._layout(data)
    self._model_layout = self._layout(model)
    self._scratch_layout = self._layout(tuple(self.arrays.items()))

  @staticmethod
  def _layout(value):
    if isinstance(value, wp.array):
      return value.ptr, value.shape, value.strides, value.dtype
    if dataclasses.is_dataclass(value):
      return tuple((field.name, _StepWorkspace._layout(getattr(value, field.name))) for field in dataclasses.fields(value))
    if isinstance(value, (tuple, list)):
      return tuple(_StepWorkspace._layout(item) for item in value)
    if value is None or isinstance(value, (int, float, bool, str)):
      return value
    return id(value)

  def validate(self, model, data):
    """Check binding before recording; GPU replay never calls this host method."""
    count, count_layout, bindings = self._execution_binding
    if (
      self.world_live_count is not count or self._layout(self.world_live_count) != count_layout or self.bindings is not bindings
    ):
      raise ValueError("Prepared execution binding changed; restore its original count and bindings before recording")
    if self.bindings is not None:
      _validate_bindings(self.bindings)
      if _binding_layout(self.bindings) != self._binding_layout:
        raise ValueError("Prepared native storage or count descriptors changed; prepare a new workspace")
    if model is not self.model or data is not self.data:
      raise ValueError("Prepared workspace requires its original model, data and immutable step options")
    if self._layout(data) != self._data_layout or self._layout(model) != self._model_layout:
      raise ValueError("Prepared Model/Data descriptors or scalar metadata changed; prepare a new workspace")
    if self._layout(tuple(self.arrays.items())) != self._scratch_layout:
      raise ValueError("Prepared scratch descriptors changed; prepare a new workspace")
    graph = self.device.captures.get(wp.get_stream(self.device))
    if graph is not None:
      owners = getattr(graph, "mjw_workspaces", ())
      if self not in owners:
        graph.mjw_workspaces = (*owners, self)

  def bind_launch(self, kernel, dim, extent_domain, *, extent_axis=0, parameter_domains=None):
    """Publish explicit count semantics after a native launch, during preparation.

    Dim only distinguishes an emitted launch from a zero-sized operation; it does
    not identify a population domain. StepBindings owns CUDA node bindings. Model,
    Data and this workspace already retain all allocation owners of the step.
    """
    if self.bindings is None:
      return
    dimensions = (dim,) if isinstance(dim, int) else tuple(dim)
    if any(size == 0 for size in dimensions):
      return
    bind_step_launch(self.bindings, kernel, dim, extent_domain, extent_axis=extent_axis, parameter_domains=parameter_domains)

  def fill(self, array, value, domain):
    """Fill an explicitly declared row domain, bounded by its native count source."""
    if array.size:
      if self.bindings is None:
        array.fill_(value)
      else:
        _fill_step_rows(self.bindings, array, value, domain)
    return array

  def copy(self, destination, source, domain):
    """Copy an explicitly declared row domain without touching inactive rows."""
    if destination.size:
      if self.bindings is None:
        wp.copy(destination, source)
      else:
        _copy_step_rows(self.bindings, destination, source, domain)

  def memory_report(self):
    return {
      "physical_scratch_bytes": None if self.storage is None else self.storage.capacity,
      "allocation_owner": "caller" if self.storage is None else "workspace",
      "payload_bytes": sum(row["payload_bytes"] for row in self._ledger),
      "fields": [{**row, "dtype": str(row["dtype"]), "pointer": self.arrays[row["name"]].ptr} for row in self._ledger],
      "scope": "scratch payload only; caller-owned physical backing and Model/Data/Contact/graph/context are excluded",
    }


def step_workspace_layout(
  model: types.Model, data: types.Data, *, world_capacity=None, contact_capacity=None, ccd_capacity=None
) -> tuple[WorkspaceFieldSpec, ...]:
  """Describe scratch before allocation for Newton/implicit-fast keyboard execution.

  Admits native NxN contacts, sleeping, pyramidal Newton and no optional callbacks,
  sensors, cameras, lights, flex, tendons, fluid or SDF. Unsupported features fail
  before allocation. Specialized free-body implicit solves are unsupported unless
  actuation, springs and dampers are all disabled, so that solve is not executed.
  Enabled ball limits, surface velocity, passive adhesion and postconstraint dynamics are
  unsupported. Dense full Jacobians above 50 padded DOFs and derivative-enabled
  gathered/sparse inertia factorizations also lack prepared count bindings.
  At least one dynamic tree is required; the static-only island path is unbound.
  Field domains separate world, candidate, CCD and scalar-counter capacity.
  Every admitted runtime launch and world/candidate/CCD memory operation must
  declare its native count domain. Route copies/fills through the workspace and reject
  unsupported execution branches here before allocating any scratch.
  A real one-world CPU or GPU Data template supplies topology/solver dimensions.
  Capacity overrides plan larger reservations without cloning Data or allocating
  storage. Omitted capacities use Data's current values. All capacities are positive
  int32 counts. This operation performs no stream or device-context operation.
  """
  m, d = model, data
  required = (
    m.opt.solver == types.SolverType.NEWTON,
    m.opt.integrator == types.IntegratorType.IMPLICITFAST,
    m.opt.cone == types.ConeType.PYRAMIDAL,
    m.opt.broadphase == types.BroadphaseType.NXN,
    bool(m.opt.enableflags & types.EnableBit.SLEEP),
    not bool(m.opt.enableflags & types.EnableBit.ENERGY),
    not bool(m.opt.disableflags & types.DisableBit.ISLAND),
    bool(m.opt.disableflags & types.DisableBit.MULTICCD),
    m.opt.run_collision_detection,
    m.opt.graph_conditional,
    not m.has_sdf_geom,
    not m.has_fluid,
  )
  absent = (m.nflex, m.ntendon, m.nsensor, m.ncam, m.nlight, m.neq, m.nacttrnbody, m.nhfield, m.na, m.nhistory)
  if not all(required) or any(absent) or any(getattr(m.callback, f.name) is not None for f in dataclasses.fields(m.callback)):
    raise NotImplementedError("Prepared workspace supports native NxN sleeping Newton/implicit-fast keyboard features only")
  if m.ntree == 0:
    raise NotImplementedError("Prepared workspace requires at least one dynamic tree")
  # The specialized free-body implicit solve has no prepared count bindings.
  derivative_flags = types.DisableBit.ACTUATION | types.DisableBit.SPRING | types.DisableBit.DAMPER
  if m.body_freeadr.size and (m.opt.disableflags & derivative_flags) != derivative_flags:
    raise NotImplementedError("Prepared workspace does not support implicit-fast free-body solves")
  if not (m.opt.disableflags & types.DisableBit.CONSTRAINT):
    if m.jnt_limited_ball_adr.size and not (m.opt.disableflags & types.DisableBit.LIMIT):
      raise NotImplementedError("Prepared workspace does not support enabled ball-joint limits")
    if m.flg_surfacevel and not (m.opt.disableflags & types.DisableBit.CONTACT):
      raise NotImplementedError("Prepared workspace does not support enabled contact surface velocity")
  if m.opt.run_rne_postconstraint:
    raise NotImplementedError("Prepared workspace does not support postconstraint inverse dynamics")
  passive_flags = types.DisableBit.SPRING | types.DisableBit.DAMPER
  if (
    m.flg_adhesion
    and m.nv > 0
    and not (m.opt.disableflags & types.DisableBit.CONTACT)
    and (m.opt.disableflags & passive_flags) != passive_flags
  ):
    raise NotImplementedError("Prepared workspace does not support enabled passive adhesion")
  if not m.is_sparse and d.nvmax_pad > 50:
    raise NotImplementedError("Prepared workspace does not support dense full Jacobians above 50 padded DOFs")
  if (m.opt.disableflags & derivative_flags) != derivative_flags and (
    any(tile.elemid.size for tile in m.M_tiles) or d.qLD.shape[1] > m.qLD_block_total
  ):
    raise NotImplementedError("Prepared workspace does not support gathered/sparse implicit inertia factorizations")
  if d.nworld < 1 or d.nvmax != m.nv:
    raise ValueError("Prepared workspace requires positive capacity and complete compact-DOF storage")
  if (
    d.dof_islandid.shape != (d.nworld, m.nv)
    or d.efc_islandid.shape != (d.nworld, d.njmax)
    or d.island_idofadr.shape != (d.nworld, m.ntree)
  ):
    raise ValueError("Prepare complete island Data before constructing a step workspace")
  m2, d2 = solver._compact_solver_views(m, d)
  table = MJ_COLLISION_TABLE.copy()
  if m.opt.disableflags & types.DisableBit.NATIVECCD:
    table[(types.GeomType.BOX, types.GeomType.BOX)] = CollisionType.PRIMITIVE
  pairs = [pair for pair, kind in table.items() if kind == CollisionType.CONVEX]
  pair_count = lambda pair: m.geom_pair_type_count[upper_trid_index(len(types.GeomType), pair[0].value, pair[1].value)]
  box = (types.GeomType.BOX, types.GeomType.BOX)
  boxes = pair_count(box) if box in pairs else 0
  iterations = 16 if boxes == sum(pair_count(pair) for pair in pairs) else m.opt.ccd_iterations
  polygon, degree = (4, 3) if boxes > 0 else (0, 0)
  nw = d.nworld if world_capacity is None else world_capacity
  candidates = d.naconmax if contact_capacity is None else contact_capacity
  nc = d.naccdmax if ccd_capacity is None else ccd_capacity
  if any(type(value) is not int or not 1 <= value < 2**31 for value in (nw, candidates, nc)):
    raise ValueError("World, contact and CCD capacities must be positive int32 counts")
  specs = [
    ("collision_pair", (candidates,), wp.vec2i, "candidate"),
    ("collision_pairid", (candidates,), wp.vec2i, "candidate"),
    ("collision_worldid", (candidates,), int, "candidate"),
    ("nccd", (len(types.GeomType) * (len(types.GeomType) + 1) // 2,), int, "global_counter"),
    ("epa_vert", (nc, 10 + 2 * iterations), wp.vec3, "ccd"),
    ("epa_vert_index", (nc, 10 + 2 * iterations), int, "ccd"),
    ("epa_face", (nc, 6 + types.MJ_MAX_EPAFACES * iterations), int, "ccd"),
    ("epa_pr", (nc, 6 + types.MJ_MAX_EPAFACES * iterations), wp.vec3, "ccd"),
    ("epa_norm2", (nc, 6 + types.MJ_MAX_EPAFACES * iterations), float, "ccd"),
    ("epa_horizon", (nc, types.MJ_MAX_EPAHORIZON), int, "ccd"),
    ("multiccd_polygon", (nc, 2 * polygon), wp.vec3, "ccd"),
    ("multiccd_clipped", (nc, 2 * polygon), wp.vec3, "ccd"),
    ("multiccd_pnormal", (nc, polygon), wp.vec3, "ccd"),
    ("multiccd_pdist", (nc, polygon), float, "ccd"),
    ("multiccd_idx1", (nc, degree), int, "ccd"),
    ("multiccd_idx2", (nc, degree), int, "ccd"),
    ("multiccd_n1", (nc, degree), wp.vec3, "ccd"),
    ("multiccd_n2", (nc, degree), wp.vec3, "ccd"),
    ("multiccd_endvert", (nc, degree), wp.vec3, "ccd"),
    ("multiccd_face1", (nc, polygon), wp.vec3, "ccd"),
    ("multiccd_face2", (nc, polygon), wp.vec3, "ccd"),
    ("awake_prev", d.body_awake.shape, int, "world"),
    ("awake_changed", (1,), int, "global_counter"),
    ("efc_nnz", (nw,), int, "world"),
    ("island_parent", (nw, m.ntree), int, "world"),
    ("island_can_sleep", (nw, m.ntree), int, "world"),
    ("efc_tree", (nw, d.njmax), int, "world"),
    ("moment_nnz", (nw,), int, "world"),
    ("nsolving", (1,), int, "global_counter"),
    ("qDeriv", (nw, m.nC), float, "world"),
    ("qLD", d.qLD.shape, float, "world"),
    ("qLDiagInv", (nw, m.nv), float, "world"),
    ("qacc", (nw, m.nv), float, "world"),
    ("actuator_vel", (nw, m.nactuator), float, "world"),
  ]
  specs.extend(("solver." + name, shape, dtype, "world") for name, shape, dtype, _ in solver._solver_context_layout(m2, d2))
  fields = []
  for name, shape, dtype, domain in specs:
    if domain == "world" and shape[0]:
      shape = (nw, *shape[1:])
    fields.append(WorkspaceFieldSpec(name, shape, {int: wp.int32, float: wp.float32, bool: wp.bool}.get(dtype, dtype), domain))
  return tuple(fields)


def make_step_workspace(model: types.Model, data: types.Data, *, arrays=None, bindings=None) -> _StepWorkspace:
  """Prepare a workspace with caller-owned typed scratch or fixed scratch allocated here.

  This is the sole public workspace construction operation. The returned owner
  records bounded scratch operations and retains all borrowed native stage views;
  its implementation type and derived views are private.

  Supplied arrays must cover step_workspace_layout exactly and retain their backing
  owner. Fields must not overlap, and all required rows must be physically ready
  before step execution. The caller owns mapping and budget policy. Inner field
  dimensions stay contiguous; outer row strides may describe packed reservations.
  Blocked Cholesky matrices (Data.cM/cqLD and solver.h/hfactor on the blocked
  Newton path) require 16-byte-aligned bases and both world and matrix-row strides.
  Scalar counters and nonblocked/empty matrices retain their natural alignment.
  Fixed execution supplies bindings=None. Dynamic execution borrows StepBindings;
  its world protected count is the sole world-live-count source. Admission enforces
  values in [0, data.nworld] and physically ready rows; candidate and CCD extents use
  their independent storage ready counts. Native operations bind declared launches
  and bound row operations in the caller's graph program. Keep the borrowed storage
  owners alive through graph retirement. Binding identity, storage, count descriptors
  and capacities are fixed at preparation. Set bindings.updates before recording;
  the updater identity is fixed at the first recorded operation. Temporarily detached
  bindings must be restored before recording any step. Only declared stage sites
  invoke binding operations; no global Warp dispatch is replaced.
  """
  if not data.qpos.device.is_cuda:
    raise ValueError("Prepared step workspace currently requires CUDA")
  if wp.get_stream(data.qpos.device).is_capturing:
    raise RuntimeError("Prepare native step workspace before graph capture")
  if bindings is not None:
    _validate_bindings(bindings)
    if bindings.world_storage.device != data.qpos.device:
      raise ValueError("Native storage must use the Data device")
    for storage, name in (
      (bindings.world_storage, "nworld"),
      (bindings.contact_storage, "naconmax"),
      (bindings.ccd_storage, "naccdmax"),
    ):
      if storage.capacity != getattr(data, name):
        raise ValueError(f"Native storage capacity must equal Data.{name}")
  specs = step_workspace_layout(model, data)
  for name in ("cM", "cqLD"):
    array = getattr(data, name)
    # The compact smooth solve always uses explicitly aligned blocked matrices.
    if array.size and (array.ptr % 16 or array.strides[0] % 16 or array.strides[1] % 16):
      raise ValueError(f"Blocked Cholesky matrix needs 16-byte base and row strides: Data.{name}")
  m2, d2 = solver._compact_solver_views(model, data)
  return _StepWorkspace(model, data, specs, m2, d2, arrays=arrays, bindings=bindings)
