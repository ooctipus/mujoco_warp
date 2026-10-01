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

from mujoco_warp._src import solver
from mujoco_warp._src import step_execution
from mujoco_warp._src import types
from mujoco_warp._src.collision_convex import _convex_scratch_layout
from mujoco_warp._src.collision_convex import _ConvexScratch
from mujoco_warp._src.collision_core import CollisionContext
from mujoco_warp._src.collision_driver import MJ_COLLISION_TABLE
from mujoco_warp._src.types import CollisionType


@dataclasses.dataclass(frozen=True)
class WorkspaceFieldSpec:
  """Scratch requirement; capacity_domain declares its independent capacity axis."""

  name: str
  shape: tuple[int, ...]
  dtype: type
  capacity_domain: str


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
    self._binding_layout = step_execution._binding_layout(bindings)
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
    self._convex = _ConvexScratch(**{field.name: self.arrays[field.name] for field in dataclasses.fields(_ConvexScratch)})
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
      step_execution._validate_bindings(self.bindings)
      if step_execution._binding_layout(self.bindings) != self._binding_layout:
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
    step_execution.bind_step_launch(
      self.bindings, kernel, dim, extent_domain, extent_axis=extent_axis, parameter_domains=parameter_domains
    )

  def fill(self, array, value, domain):
    """Fill an explicitly declared row domain, bounded by its native count source."""
    if array.size:
      if self.bindings is None:
        array.fill_(value)
      else:
        step_execution._fill_step_rows(self.bindings, array, value, domain)
    return array

  def copy(self, destination, source, domain):
    """Copy an explicitly declared row domain without touching inactive rows."""
    if destination.size:
      if self.bindings is None:
        wp.copy(destination, source)
      else:
        step_execution._copy_step_rows(self.bindings, destination, source, domain)

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
  nw = d.nworld if world_capacity is None else world_capacity
  candidates = d.naconmax if contact_capacity is None else contact_capacity
  nc = d.naccdmax if ccd_capacity is None else ccd_capacity
  if any(type(value) is not int or not 1 <= value < 2**31 for value in (nw, candidates, nc)):
    raise ValueError("World, contact and CCD capacities must be positive int32 counts")
  specs = [
    ("collision_pair", (candidates,), wp.vec2i, "candidate"),
    ("collision_pairid", (candidates,), wp.vec2i, "candidate"),
    ("collision_worldid", (candidates,), int, "candidate"),
    *_convex_scratch_layout(m, pairs, nc)[2],
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
    step_execution._validate_bindings(bindings)
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
