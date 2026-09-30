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
class WorkspaceField:
  """One semantic scratch field; domain declares its independent capacity axis."""

  name: str
  shape: tuple[int, ...]
  dtype: type
  domain: str


class StepWorkspace:
  """Own one fixed scratch allocation and borrowed native stage views.

  Prepare with make_step_workspace. One workspace may serve ordered substeps and
  both collision passes; concurrent steps require separate workspaces. Model and
  Data storage must remain unchanged. Captured steps retain this owner on the graph.
  """

  def __init__(self, model, data, live_count, specs, solver_model, solver_data, arrays=None, observer=None):
    self.model, self.data, self.live_count = model, data, live_count
    self.observer = observer
    self.device = data.qpos.device
    self.solver_model, self.solver_data = solver_model, solver_data
    self.arrays, self._ledger = {}, []
    offset = 0
    for spec in specs:
      name, shape, dtype, scope = spec.name, spec.shape, spec.dtype, spec.domain
      nbytes = math.prod(shape) * wp.types.type_size_in_bytes(dtype)
      offset = (offset + 127) // 128 * 128
      field = dict(name=name, shape=shape, dtype=dtype, offset=offset, bytes=nbytes, scope=scope)
      field["world_axis"] = 0 if scope == "world" else None
      self._ledger.append(field)
      offset += nbytes
    self.storage = None
    if arrays is None:
      self.storage = wp.empty(offset, dtype=wp.uint8, device=self.device)
      for field in self._ledger:
        array = (
          wp.array(ptr=self.storage.ptr + field["offset"], shape=field["shape"], dtype=field["dtype"], device=self.device)
          if field["bytes"]
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
        self.arrays[spec.name] = array
        self._ledger[len(self.arrays) - 1]["offset"] = None
    self.collision = CollisionContext(
      **{name: self.arrays[name] for name in ("collision_pair", "collision_pairid", "collision_worldid")}
    )
    self.solver_context = types.SolverContext(
      **{name: self.arrays["solver." + name] for name, _, _, _ in solver._solver_context_layout(solver_model, solver_data)}
    )
    self.solver_context.compact_m_full = model
    self.solver_context.compact_d_full = data
    self._data_layout = self._layout(data)
    self._model_layout = self._layout(model)
    self._scratch_layout = self._layout(tuple(self.arrays.items()))

  @staticmethod
  def _layout(value):
    if isinstance(value, wp.array):
      return value.ptr, value.shape, value.strides, value.dtype
    if dataclasses.is_dataclass(value):
      return tuple((field.name, StepWorkspace._layout(getattr(value, field.name))) for field in dataclasses.fields(value))
    if isinstance(value, tuple):
      return tuple(StepWorkspace._layout(item) for item in value)
    if value is None or isinstance(value, (int, float, bool, str)):
      return value
    return id(value)

  def validate(self, model, data):
    """Check binding before recording; GPU replay never calls this host method."""
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

  def world_arrays(self):
    """Return declared nonempty world scratch; no numerical extent inference."""
    return {field["name"]: self.arrays[field["name"]] for field in self._ledger if field["world_axis"] == 0 and field["bytes"]}

  def observe_launch(self, kernel, dim, extent_domain, *, extent_axis=0, parameters=None):
    """Publish explicit count semantics after a native launch, during preparation.

    Dim only distinguishes an emitted launch from a zero-sized operation; it does
    not identify a population domain. The observer owns CUDA node bindings. Model,
    Data and this workspace already retain all allocation owners of the step.
    """
    if self.observer is None:
      return
    dimensions = (dim,) if isinstance(dim, int) else tuple(dim)
    if any(size == 0 for size in dimensions):
      return
    self.observer.observe_launch(kernel, dim, extent_domain, extent_axis=extent_axis, parameters=parameters or {})

  def fill(self, array, value, domain):
    """Fill an explicitly declared row domain, bounded by the observer's count."""
    if array.size:
      if self.observer is None:
        array.fill_(value)
      else:
        self.observer.fill(array, value, domain)
    return array

  def copy(self, destination, source, domain):
    """Copy an explicitly declared row domain without touching inactive rows."""
    if destination.size:
      if self.observer is None:
        wp.copy(destination, source)
      else:
        self.observer.copy(destination, source, domain)

  def memory_report(self):
    return {
      "physical_scratch_bytes": None if self.storage is None else self.storage.capacity,
      "allocation_owner": "caller" if self.storage is None else "workspace",
      "payload_bytes": sum(row["bytes"] for row in self._ledger),
      "fields": [{**row, "dtype": str(row["dtype"]), "pointer": self.arrays[row["name"]].ptr} for row in self._ledger],
      "scope": "scratch payload only; caller-owned physical backing and Model/Data/Contact/graph/context are excluded",
    }


def step_workspace_layout(
  model: types.Model, data: types.Data, *, world_capacity=None, contact_capacity=None, ccd_capacity=None
) -> tuple[WorkspaceField, ...]:
  """Describe scratch before allocation for Newton/implicit-fast keyboard execution.

  Admits native NxN contacts, sleeping, pyramidal Newton and no optional callbacks,
  sensors, flex, tendons, fluid or SDF. Unsupported features fail before allocation.
  Field domains separate world, candidate, CCD and scalar-counter capacity.
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
    not bool(m.opt.disableflags & types.DisableBit.ISLAND),
    bool(m.opt.disableflags & types.DisableBit.MULTICCD),
    m.opt.run_collision_detection,
    m.opt.graph_conditional,
    not m.has_sdf_geom,
    not m.has_fluid,
  )
  absent = (m.nflex, m.ntendon, m.nsensor, m.neq, m.nacttrnbody, m.nhfield, m.na, m.nhistory)
  if not all(required) or any(absent) or any(getattr(m.callback, f.name) is not None for f in dataclasses.fields(m.callback)):
    raise NotImplementedError("Prepared workspace supports native NxN sleeping Newton/implicit-fast keyboard features only")
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
    fields.append(WorkspaceField(name, shape, {int: wp.int32, float: wp.float32, bool: wp.bool}.get(dtype, dtype), domain))
  return tuple(fields)


def make_step_workspace(model: types.Model, data: types.Data, *, live_count=None, arrays=None, observer=None) -> StepWorkspace:
  """Bind complete caller-owned typed scratch, or allocate fixed scratch by default.

  Supplied arrays must cover step_workspace_layout exactly and retain their backing
  owner. Fields must not overlap, and all required rows must be physically ready
  before step execution. The caller owns mapping and budget policy. Inner field
  dimensions stay contiguous; outer row strides may describe packed reservations.
  A live_count is a CUDA int32 scalar in [0, data.nworld], enforced by admission.
  An optional observer implements observe_launch, fill and copy to prepare explicit
  launch-count bindings and bounded row operations in the caller's graph program.
  It is invoked only at declared stage sites; no global Warp dispatch is replaced.
  """
  if not data.qpos.device.is_cuda:
    raise ValueError("Prepared step workspace currently requires CUDA")
  if wp.get_stream(data.qpos.device).is_capturing:
    raise RuntimeError("Prepare native step workspace before graph capture")
  if observer is not None and not all(callable(getattr(observer, name, None)) for name in ("observe_launch", "fill", "copy")):
    raise TypeError("Prepared observer must implement observe_launch, fill and copy")
  specs = step_workspace_layout(model, data)
  if live_count is not None and (
    not isinstance(live_count, wp.array)
    or live_count.dtype != wp.int32
    or live_count.shape != (1,)
    or not live_count.is_contiguous
    or live_count.device != data.qpos.device
  ):
    raise ValueError("Live count must be one contiguous int32 scalar on the Data device")
  m2, d2 = solver._compact_solver_views(model, data)
  return StepWorkspace(model, data, live_count, specs, m2, d2, arrays=arrays, observer=observer)
