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


class _StepWorkspace:
  """Internal owner of prepared scratch and borrowed native stage views.

  Prepare with make_step_workspace. One workspace may serve ordered substeps and
  both collision passes; concurrent steps require separate workspaces. Model and
  Data storage must remain unchanged. Captured steps retain this owner on the graph.
  Compact solver and collision views are private engine implementation; callers
  provide only the original Model/Data and declared scratch descriptors.
  """

  def __init__(self, model, data, world_live_count, specs, solver_model, solver_data, arrays=None, recorder=None):
    self.model, self.data, self.world_live_count = model, data, world_live_count
    self.recorder = recorder
    self._execution_binding = world_live_count, self._layout(world_live_count), recorder
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
    _validate_execution_binding(self.world_live_count, self.recorder)
    count, count_layout, recorder = self._execution_binding
    if (
      self.world_live_count is not count or self._layout(self.world_live_count) != count_layout or self.recorder is not recorder
    ):
      raise ValueError("Prepared execution binding changed; restore its original world count and recorder before recording")
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
    not identify a population domain. The recorder owns CUDA node bindings. Model,
    Data and this workspace already retain all allocation owners of the step.
    """
    if self.recorder is None:
      return
    dimensions = (dim,) if isinstance(dim, int) else tuple(dim)
    if any(size == 0 for size in dimensions):
      return
    self.recorder.bind_launch(kernel, dim, extent_domain, extent_axis=extent_axis, parameter_domains=parameter_domains or {})

  def fill(self, array, value, domain):
    """Fill an explicitly declared row domain, bounded by the recorder's count."""
    if array.size:
      if self.recorder is None:
        array.fill_(value)
      else:
        self.recorder.fill(array, value, domain)
    return array

  def copy(self, destination, source, domain):
    """Copy an explicitly declared row domain without touching inactive rows."""
    if destination.size:
      if self.recorder is None:
        wp.copy(destination, source)
      else:
        self.recorder.copy(destination, source, domain)

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
  declare its recorder domain. Route copies/fills through the workspace and reject
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


def _validate_execution_binding(world_live_count, recorder):
  if recorder is not None and not all(callable(getattr(recorder, name, None)) for name in ("bind_launch", "fill", "copy")):
    raise TypeError("Prepared recorder must implement bind_launch, fill and copy")
  if (world_live_count is None) != (recorder is None):
    raise ValueError("Dynamic execution requires world_live_count and recorder together; fixed execution supplies neither")


def make_step_workspace(
  model: types.Model, data: types.Data, *, world_live_count=None, arrays=None, recorder=None
) -> _StepWorkspace:
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
  Fixed execution supplies neither world_live_count nor recorder. Dynamic execution
  supplies both: world_live_count is a CUDA int32 scalar in [0, data.nworld], enforced
  by admission, and recorder implements bind_launch, fill and copy. The recorder
  must bind every declared world launch to that same count, with independent
  candidate/CCD counts, and bound row operations in the caller's graph program.
  The count descriptor, execution mode and recorder identity are fixed at preparation.
  A temporarily detached recorder must be restored before recording any step.
  The recorder runs only at declared
  stage sites; no global Warp dispatch is replaced.
  """
  if not data.qpos.device.is_cuda:
    raise ValueError("Prepared step workspace currently requires CUDA")
  if wp.get_stream(data.qpos.device).is_capturing:
    raise RuntimeError("Prepare native step workspace before graph capture")
  _validate_execution_binding(world_live_count, recorder)
  specs = step_workspace_layout(model, data)
  for name in ("cM", "cqLD"):
    array = getattr(data, name)
    # The compact smooth solve always uses explicitly aligned blocked matrices.
    if array.size and (array.ptr % 16 or array.strides[0] % 16 or array.strides[1] % 16):
      raise ValueError(f"Blocked Cholesky matrix needs 16-byte base and row strides: Data.{name}")
  if world_live_count is not None and (
    not isinstance(world_live_count, wp.array)
    or world_live_count.dtype != wp.int32
    or world_live_count.shape != (1,)
    or not world_live_count.is_contiguous
    or world_live_count.device != data.qpos.device
  ):
    raise ValueError("World live count must be one contiguous int32 scalar on the Data device")
  m2, d2 = solver._compact_solver_views(model, data)
  return _StepWorkspace(model, data, world_live_count, specs, m2, d2, arrays=arrays, recorder=recorder)
