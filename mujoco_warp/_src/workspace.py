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

"""Discover numerical scratch, then bind one complete prepared execution."""

import dataclasses
import traceback

import warp as wp
from gpu_components import fields as field_ops
from gpu_components import graph as graph_ops
from gpu_components import scratch as scratch_ops
from gpu_components import scratch_data

from mujoco_warp._src import forward
from mujoco_warp._src import io
from mujoco_warp._src import solver
from mujoco_warp._src import step_program
from mujoco_warp._src import types


@dataclasses.dataclass(eq=False)
class _StepWorkspace:
  """Borrow the complete native execution and its immutable resource relationship."""

  model: types.Model
  data: types.Data
  execution_data: types.Data
  bindings: step_program.StepBindings
  device: object
  _layout: tuple
  _binding_layout: tuple


def validate_step_model(model: types.Model, data: types.Data) -> None:
  """Reject features without prepared execution and initialization contracts."""
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


def discover_step_scratch(model, template, *, world_capacity=None, contact_capacity=None, ccd_capacity=None):
  """Discover native scratch before allocating the final capacity domains.

  Return a frozen registry and the exact ``(nworld, naconmax, naccdmax)`` count
  parameters that the final workspace must reuse. A discarded recording uses
  capacity-sized graph allocations for Data descriptors; it never executes or
  initializes simulation payload. The caller warms the native program first.
  """
  validate_step_model(model, template)
  if template.scratch is not None:
    raise ValueError("Scratch discovery requires an unbound native Data template")
  capacities = (
    template.nworld if world_capacity is None else world_capacity,
    template.naconmax if contact_capacity is None else contact_capacity,
    template.naccdmax if ccd_capacity is None else ccd_capacity,
  )
  if any(type(value) is not int or not 1 <= value < 2**31 for value in capacities):
    raise ValueError("Discovery capacities must be positive int32 counts")
  device = template.qpos.device
  if not device.is_cuda or device.is_capturing:
    raise RuntimeError("Scratch discovery requires a CUDA device outside capture")
  if not wp.is_mempool_enabled(device):
    raise RuntimeError("Scratch discovery requires the CUDA memory pool")
  counts = tuple(wp.CountParameter(value) for value in capacities)
  capacity_by_axis = dict(zip(("nworld", "naconmax", "naccdmax"), capacities, strict=True))
  registry = scratch_ops.allocate_registry(device)
  arrays, execution, failure = {}, None, None
  capture = wp.ScopedCapture(
    device=device,
    force_module_load=False,
    capture_mode=wp.CaptureMode.THREAD_LOCAL,
    record_launches=True,
    record_memory_operations=True,
  )
  try:
    with wp.ScopedDevice(device), capture:
      for name, array, declared_shape in io.array_fields(template):
        axis = declared_shape[0] if declared_shape else None
        if axis in capacity_by_axis and array.shape[0]:
          arrays[name] = wp.empty((capacity_by_axis[axis], *array.shape[1:]), dtype=array.dtype, device=device)
      execution = dataclasses.replace(
        io.replace_arrays(template, arrays), nworld=counts[0], naconmax=counts[1], naccdmax=counts[2], scratch=registry
      )
      forward.step(model, execution)
  except BaseException as error:
    failure = error
    traceback.clear_frames(error.__traceback__)
  try:
    if capture.graph is not None:
      graph_ops.invalidate(capture.graph)
  except BaseException as cleanup:
    if failure is not None:
      raise BaseExceptionGroup("Scratch discovery and graph invalidation failed", [failure, cleanup]) from failure
    raise
  finally:
    execution = capture = None
    arrays.clear()
    scratch_ops.freeze(registry)
  if failure is not None:
    raise failure
  return registry, counts


def make_step_workspace(model, data, *, bindings, scratch, count_parameters) -> _StepWorkspace:
  """Bind complete scratch and state storage once, before recording execution.

  Discovery supplies three count identities in world, candidate, CCD order. Newton
  prepares all columns in their corresponding domain storage before calling this
  operation. No prepared object exists while those relationships are incomplete.
  Ordinary numerical execution uses Data.scratch=None and needs no workspace.
  """
  if not data.qpos.device.is_cuda or wp.get_stream(data.qpos.device).is_capturing:
    raise RuntimeError("Prepare a step workspace on CUDA before graph capture")
  validate_step_model(model, data)
  step_program._validate_bindings(bindings)
  if scratch.state != scratch_data.PREPARED or scratch.device != data.qpos.device:
    raise ValueError("Prepare scratch on the Data device before binding the step")
  if len(count_parameters) != 3 or len({id(count) for count in count_parameters}) != 3:
    raise ValueError("Prepared execution needs three distinct count parameters")
  storages = (bindings.world_storage, bindings.contact_storage, bindings.ccd_storage)
  for name, parameter, storage in zip(("nworld", "naconmax", "naccdmax"), count_parameters, storages):
    if not isinstance(parameter, wp.CountParameter) or parameter.maximum != getattr(data, name):
      raise ValueError(f"Prepared count must match Data.{name}")
    if storage.capacity != parameter.maximum or storage.device != data.qpos.device:
      raise ValueError(f"Prepared storage must match Data.{name} and its device")
  execution = dataclasses.replace(data, **dict(zip(("nworld", "naconmax", "naccdmax"), count_parameters)), scratch=scratch)
  if scratch.columns.keys() != scratch.declarations.keys() or scratch.temporaries:
    raise ValueError("Prepared scratch must cover its declarations exactly, without discovery temporaries")
  for name, declaration in scratch.declarations.items():
    column = scratch.columns[name]
    if (column.shape, column.dtype, column.device) != (
      scratch_ops.bounded_shape(declaration.shape),
      declaration.dtype,
      scratch.device,
    ):
      raise ValueError(f"Scratch column {name!r} does not match its declaration")
    if isinstance(declaration.shape[0], wp.CountParameter):
      owner = next((s for p, s in zip(count_parameters, storages) if p is declaration.shape[0]), None)
      if owner is None or (column.size and field_ops.lookup(owner, column) is None):
        raise ValueError(f"Scratch column {name!r} must belong to its declared capacity domain")
    elif not any(column is array for array in bindings.fixed_arrays):
      raise ValueError(f"Count-free scratch {name!r} must be a bound fixed array")
  field_ops.validate_disjoint_arrays(
    (
      *scratch.columns.values(),
      *(a for a in bindings.fixed_arrays if all(a is not column for column in scratch.columns.values())),
      *(c for _, c in step_program._count_sources(execution, bindings)),
    )
  )
  # Native Data may legitimately alias its own views; temporaries must never alias it.
  field_ops.validate_disjoint_arrays(scratch.columns.values(), against=(array for _, array, _ in io.array_fields(data)))
  solver.validate_blocked_matrix(data.cM)
  solver.validate_blocked_matrix(data.cqLD)
  return _StepWorkspace(
    model,
    data,
    execution,
    bindings,
    data.qpos.device,
    step_program._layout((model, data, execution)),
    step_program._binding_layout(bindings),
  )


def step_workspace_memory_report(workspace):
  """Report borrowed scratch payload; physical storage is reported by its owner."""
  return {
    "allocation_owner": "caller",
    "payload_bytes": sum(
      a.size * wp.types.type_size_in_bytes(a.dtype) for a in workspace.execution_data.scratch.columns.values()
    ),
    "scope": "borrowed scratch payload only",
  }
