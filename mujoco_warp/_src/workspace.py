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
from typing import get_type_hints

import warp as wp
from gpu_components import fields as field_ops

from mujoco_warp._src import solver
from mujoco_warp._src import step_execution
from mujoco_warp._src import types
from mujoco_warp._src.collision_convex import _convex_scratch_shapes
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


@dataclasses.dataclass(eq=False)
class _StepWorkspace:
  """Passive prepared scratch and borrowed stage views, constructed by make_step_workspace.

  One record serves ordered substeps and collision passes. Concurrent steps need
  separate scratch. Only preparation operations create owners and derived views;
  native execution operations validate bindings and retain this record on capture.
  """

  model: types.Model
  data: types.Data
  bindings: step_execution.StepBindings | None
  device: object
  storage: wp.array | None
  arrays: dict[str, wp.array]
  convex: _ConvexScratch
  _collision: CollisionContext
  _solver_model: object
  _solver_data: object
  _solver_context: types.SolverContext
  _ledger: tuple[dict, ...]
  _execution_binding: object
  _binding_layout: tuple | None
  _data_layout: object
  _model_layout: object
  _scratch_layout: object


def step_workspace_memory_report(workspace):
  """Describe scratch payload without claiming ownership of caller-managed backing."""
  return {
    "physical_scratch_bytes": None if workspace.storage is None else workspace.storage.capacity,
    "allocation_owner": "caller" if workspace.storage is None else "workspace",
    "payload_bytes": sum(field["payload_bytes"] for field in workspace._ledger),
    "fields": [
      {**field, "dtype": str(field["dtype"]), "pointer": workspace.arrays[field["name"]].ptr} for field in workspace._ledger
    ],
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
  _, _, convex_shapes = _convex_scratch_shapes(m, pairs, nc)
  convex_types = get_type_hints(_ConvexScratch)
  if set(convex_shapes) != set(convex_types) or any(
    len(convex_shapes[name]) != hint.ndim for name, hint in convex_types.items()
  ):
    raise ValueError("Convex shapes must match the typed scratch declaration")
  specs = [
    ("collision_pair", (candidates,), wp.vec2i, "candidate"),
    ("collision_pairid", (candidates,), wp.vec2i, "candidate"),
    ("collision_worldid", (candidates,), int, "candidate"),
    *(
      (name, convex_shapes[name], hint.dtype, "global_counter" if name == "nccd" else "ccd")
      for name, hint in convex_types.items()
    ),
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
  retains borrowed native stage views and data; free operations perform execution;
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
  if bindings is not None and arrays is None:
    raise ValueError("Dynamic execution requires caller-owned registered scratch")
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
  solver.validate_blocked_matrix(data.cM)
  solver.validate_blocked_matrix(data.cqLD)
  m2, d2 = solver._compact_solver_views(model, data)
  device = data.qpos.device
  layouts = tuple(field_ops.contiguous(spec.shape, wp.types.type_size_in_bytes(spec.dtype)) for spec in specs)
  for spec, layout in zip(specs, layouts):
    field_ops.validate_layout(layout, dtype=spec.dtype)
  spans = tuple(field_ops.span_bytes(layout, wp.types.type_size_in_bytes(spec.dtype)) for spec, layout in zip(specs, layouts))
  offsets, nbytes = field_ops.pack(spans, (128,) * len(specs), end_alignment_bytes=128)
  storage = None
  if arrays is None:
    buffer_shape = (nbytes // 128, 128)
    field_ops.validate_layout(field_ops.contiguous(buffer_shape, 1), dtype=wp.uint8)
    storage = wp.empty(buffer_shape, dtype=wp.uint8, device=device)
    if nbytes and storage.ptr % 128:
      raise ValueError("Scratch allocation requires 128-byte base alignment")
    arrays = {
      spec.name: field_ops.bind(storage, spec.dtype, layout, byte_offset=offset)
      for spec, layout, offset in zip(specs, layouts, offsets)
    }
  else:
    if set(arrays) != {spec.name for spec in specs}:
      raise ValueError("Caller-owned scratch must supply exactly the declared field names")
    arrays = dict(arrays)
    for spec in specs:
      field_ops.validate_array(arrays[spec.name], dtype=spec.dtype, shape=spec.shape, device=device)
    field_ops.validate_disjoint_arrays(tuple(arrays.values()))
    if bindings is not None:
      domains = {"world": bindings.world_storage, "candidate": bindings.contact_storage, "ccd": bindings.ccd_storage}
      for spec in specs:
        array = arrays[spec.name]
        if spec.capacity_domain in domains and spec.shape[0]:
          if field_ops.lookup(domains[spec.capacity_domain], array) is None:
            raise ValueError(f"Scratch must be registered in its declared {spec.capacity_domain} storage: {spec.name}")
        elif spec.capacity_domain == "global_counter" and not array.is_contiguous:
          raise ValueError(f"Global counters require dense descriptors and caller-guaranteed full backing: {spec.name}")
  collision = CollisionContext(**{name: arrays[name] for name in ("collision_pair", "collision_pairid", "collision_worldid")})
  convex = _ConvexScratch(**{field.name: arrays[field.name] for field in dataclasses.fields(_ConvexScratch)})
  solver_context = types.SolverContext(
    **{name: arrays["solver." + name] for name, _, _, _ in solver._solver_context_layout(m2, d2)}
  )
  solver.validate_solver_scratch(m2, solver_context)
  solver_context.compact_m_full, solver_context.compact_d_full = model, data
  ledger = tuple(
    dict(
      name=spec.name,
      shape=spec.shape,
      dtype=spec.dtype,
      allocation_offset_bytes=offset if storage is not None else None,
      payload_bytes=span,
      capacity_domain=spec.capacity_domain,
      world_axis=0 if spec.capacity_domain == "world" else None,
    )
    for spec, offset, span in zip(specs, offsets, spans)
  )
  memo = {}
  return _StepWorkspace(
    model=model,
    data=data,
    bindings=bindings,
    device=device,
    storage=storage,
    arrays=arrays,
    convex=convex,
    _collision=collision,
    _solver_model=m2,
    _solver_data=d2,
    _solver_context=solver_context,
    _ledger=ledger,
    _execution_binding=bindings,
    _binding_layout=step_execution._binding_layout(bindings),
    _data_layout=step_execution._layout(data, memo),
    _model_layout=step_execution._layout(model, memo),
    _scratch_layout=step_execution._layout((tuple(arrays.items()), convex, collision, solver_context, m2, d2), memo),
  )
