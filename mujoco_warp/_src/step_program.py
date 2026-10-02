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

"""Compile declared native count relations onto an already captured step.

Physics emits ordinary Warp launches. This composition boundary imports native
kernel definitions and factory identities, then assigns the explicit population's
count sources. It neither infers semantics from shapes nor discovers ownership
from operands. Workspace preparation owns scratch; Warp records ordinary memory
operations, and generic fields operations validate their explicit storage domains.
"""

from __future__ import annotations

import dataclasses

import warp as wp
from gpu_components import fields as field_ops
from gpu_components import graph as graph_ops
from gpu_components.field_data import FieldStorage
from gpu_components.graph_data import GraphKernelBinding
from gpu_components.graph_data import GraphUpdateTable
from gpu_components.graph_data import KernelParameterBinding

from mujoco_warp._src import collision_convex
from mujoco_warp._src import collision_driver
from mujoco_warp._src import collision_primitive
from mujoco_warp._src import constraint
from mujoco_warp._src import derivative
from mujoco_warp._src import forward
from mujoco_warp._src import island
from mujoco_warp._src import passive
from mujoco_warp._src import sleep
from mujoco_warp._src import smooth
from mujoco_warp._src import solver
from mujoco_warp._src import support
from mujoco_warp._src.warp_util import kernel_instances


@dataclasses.dataclass(eq=False)
class StepBindings:
  """Borrowed native capacity domains and their single captured binding ledger.

  Keep all three storage owners alive through graph retirement. World launches use
  world_storage.protected_count; candidate and CCD launches use their ready_count.
  Admission must keep those counts within physically ready rows. Set updates before
  the first recorded operation. Only native binding operations mutate the bindings
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
  _recording_binding: tuple | None = None


_LAYOUT_REFERENCE = object()


def _layout(value, memo=None):
  """Snapshot native metadata with local references for repeated borrowed aliases.

  Share a fresh memo across one operation's roots. First occurrences record all
  descriptors; later occurrences record their traversal ordinal. The memo retains
  temporary containers and is never reused by a subsequent validation.
  """
  if value is None or isinstance(value, (int, float, bool, str)):
    return value
  if memo is None:
    memo = {}
  identity = id(value)
  if identity in memo:
    return memo[identity][1]
  if isinstance(value, wp.array):
    memo[identity] = value, (_LAYOUT_REFERENCE, len(memo))
    return value.ptr, value.shape, value.strides, value.dtype, value.device
  if dataclasses.is_dataclass(value):
    memo[identity] = value, (_LAYOUT_REFERENCE, len(memo))
    result = []
    for field in dataclasses.fields(value):
      child = getattr(value, field.name)
      cached = memo.get(id(child))
      result.append((field.name, cached[1] if cached is not None else _layout(child, memo)))
    return tuple(result)
  if isinstance(value, (tuple, list)):
    memo[identity] = value, (_LAYOUT_REFERENCE, len(memo))
    result = []
    for child in value:
      cached = memo.get(id(child))
      result.append(cached[1] if cached is not None else _layout(child, memo))
    return tuple(result)
  return identity


def validate_step_workspace(workspace, model, data):
  """Validate prepared binding before emission and pin owners during graph recording."""
  if workspace.bindings is not workspace._execution_binding:
    raise ValueError("Prepared execution binding changed; restore its original bindings before recording")
  if workspace.bindings is not None:
    if _binding_layout(workspace.bindings) != workspace._binding_layout:
      raise ValueError("Prepared native storage or count descriptors changed; prepare a new workspace")
    _validate_bindings(workspace.bindings)
  if model is not workspace.model or data is not workspace.data:
    raise ValueError("Prepared workspace requires its original model, data and immutable step options")
  memo = {}
  if _layout(data, memo) != workspace._data_layout or _layout(model, memo) != workspace._model_layout:
    raise ValueError("Prepared Model/Data descriptors or scalar metadata changed; prepare a new workspace")
  if (
    _layout(
      (tuple(workspace.arrays.items()), workspace.scratch),
      memo,
    )
    != workspace._scratch_layout
  ):
    raise ValueError("Prepared scratch descriptors changed; prepare a new workspace")
  owner = graph_ops.current_capture(device=workspace.device)
  if owner is not None:
    try:
      if workspace.bindings is not None:
        if workspace.bindings.updates is None:
          raise RuntimeError("Prepare native graph updates before recording step operations")
        _retain_bindings(workspace.bindings, owner)
      graph_ops.retain(owner, workspace)
    except BaseException:
      if workspace.bindings is not None:
        workspace.bindings.recording_failed = True
      graph_ops.invalidate(owner)
      raise


def _binding_layout(bindings):
  if bindings is None:
    return None
  return tuple(
    (id(storage), storage.capacity, id(storage.device), id(count), (count.ptr, count.shape, count.strides, count.dtype))
    for storage, count in (
      (bindings.world_storage, bindings.world_storage.protected_count),
      (bindings.contact_storage, bindings.contact_storage.ready_count),
      (bindings.ccd_storage, bindings.ccd_storage.ready_count),
    )
  ) + (id(bindings.bindings),)


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
    raise ValueError("Native recording binding changed; restore its original storage, counts, updates and bindings")


def resolve_step_counts(bindings, kernel, extent_domain, extent_axis, parameter_domains):
  """Resolve native count labels to numeric descriptors; mutate no recording state.

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


def _retain_bindings(bindings, graph):
  """Retain admitted resources and freeze the one binding relation for this program."""
  _validate_bindings(bindings)
  if bindings._recording_binding is None:
    graph_ops.retain(graph, bindings)
    for storage in (bindings.world_storage, bindings.contact_storage, bindings.ccd_storage):
      field_ops.retain_graph(storage, graph)
    bindings._recording_binding = _binding_layout(bindings), id(bindings.updates)


def _fail_recording(bindings, stream=None):
  if not isinstance(bindings, StepBindings):
    return
  bindings.recording_failed = True
  try:
    device = (
      bindings.updates.device
      if isinstance(bindings.updates, GraphUpdateTable)
      else bindings.world_storage.protected_count.device
    )
    graph = graph_ops.current_capture(device=device, stream=stream)
    if graph is not None:
      graph_ops.invalidate(graph)
  except BaseException:
    # A malformed borrowed descriptor must not mask the error that poisoned it.
    # The composition root must also reject the native recording_failed latch.
    pass


_WORLD, _CANDIDATE, _CCD = range(3)
_DOMAIN_NAMES = ("world", "candidate", "ccd")

# Every dynamic native extent is its declared population's leading axis.
# Scalar labels are a schema front door only; compilation resolves exact int32
# argument positions before the generic graph API receives a declaration.
_KERNEL_DOMAINS = {
  collision_driver._any_awake_changed: (_WORLD, ()),
  constraint._zero_constraint_counts: (_WORLD, ()),
  derivative._qderiv_actuator_passive: (_WORLD, ()),
  derivative._qderiv_actuator_passive_actuation_sparse: (_WORLD, ()),
  derivative._qderiv_actuator_passive_vel: (_WORLD, ()),
  derivative._qderiv_tendon_damping: (_WORLD, ()),
  forward._actuator_force: (_WORLD, ()),
  forward._actuator_velocity: (_WORLD, ()),
  forward._next_activation: (_WORLD, ()),
  forward._next_position: (_WORLD, ()),
  forward._next_velocity: (_WORLD, ()),
  forward._qfrc_actuator: (_WORLD, ()),
  forward._qfrc_actuator_gravcomp_limits: (_WORLD, ()),
  island._compress_roots: (_WORLD, ()),
  island._compute_efc_tree: (_WORLD, ()),
  island._init_dof_arrays: (_WORLD, ()),
  island._init_efc_arrays: (_WORLD, ()),
  island._init_island_arrays: (_WORLD, ()),
  island._island_count_constraints: (_WORLD, ()),
  island._island_count_dofs: (_WORLD, ()),
  island._island_dsu: (_WORLD, ()),
  island._island_map_constraints: (_WORLD, ()),
  island._island_map_dofs: (_WORLD, ()),
  island._island_scan_sizes: (_WORLD, ()),
  island._label_roots: (_WORLD, ()),
  island._propagate_labels: (_WORLD, ()),
  island._reset_compact_maps: (_WORLD, ()),
  island._reset_dsu: (_WORLD, ()),
  passive._gravity_force: (_WORLD, ()),
  passive._spring_damper_dof_passive: (_WORLD, ()),
  sleep._build_cycles: (_WORLD, ()),
  sleep._check_island_can_sleep: (_WORLD, ()),
  sleep._clear_disabled_dofs: (_WORLD, ()),
  sleep._sweep_awake_trees: (_WORLD, ()),
  sleep._update_sleep_bodies: (_WORLD, ()),
  sleep._update_sleep_dofs: (_WORLD, ()),
  sleep._update_sleep_trees: (_WORLD, ()),
  sleep._wake_collision_kernel: (_CANDIDATE, ()),
  sleep._wake_kernel: (_WORLD, ()),
  sleep._zero_sleep_counters: (_WORLD, ()),
  smooth._M: (_WORLD, ()),
  smooth._cacc_branch: (_WORLD, ()),
  smooth._cacc_world: (_WORLD, ()),
  smooth._cdof: (_WORLD, ()),
  smooth._cfrc: (_WORLD, ()),
  smooth._cfrc_backward: (_WORLD, ()),
  smooth._cinert: (_WORLD, ()),
  smooth._compute_body_inertial_frames: (_WORLD, ()),
  smooth._compute_body_matrices: (_WORLD, ()),
  smooth._comvel_branch: (_WORLD, ()),
  smooth._comvel_root: (_WORLD, ()),
  smooth._crb_accumulate: (_WORLD, ()),
  smooth._geom_local_to_global: (_WORLD, ()),
  smooth._kinematics_branch: (_WORLD, ()),
  smooth._qfrc_bias: (_WORLD, ()),
  smooth._site_local_to_global: (_WORLD, ()),
  smooth._subtree_com_acc: (_WORLD, ()),
  smooth._subtree_com_init: (_WORLD, ()),
  smooth._subtree_div: (_WORLD, ()),
  smooth._transmission: (_WORLD, ()),
  solver._gather_J_dense: (_WORLD, ()),
  solver._gather_M_sparse: (_WORLD, ()),
  solver._gather_dof_vecs_compact: (_WORLD, ()),
  solver._gather_rhs_compact: (_WORLD, ()),
  solver._init_compact_inertia: (_WORLD, ()),
  solver._mul_m_sparse_compact: (_WORLD, ()),
  solver._qfrc_constraint_from_grad: (_WORLD, ()),
  solver._scatter_dof_vecs: (_WORLD, ()),
  solver._scatter_solution: (_WORLD, ()),
  solver._solve_init_efc: (_WORLD, ()),
  solver._update_gradient_h_incremental: (_WORLD, ()),
  solver._zero_change_counters: (_WORLD, ()),
  solver._zero_qfrc_constraint_sparse: (_WORLD, ()),
  support._apply_ft: (_WORLD, ()),
}

_FACTORY_DOMAINS = {
  collision_convex.ccd_kernel_builder: (None, (("naconmax_in", _CANDIDATE), ("naccdmax_in", _CCD))),
  collision_driver._nxn_broadphase: (_WORLD, (("naconmax_in", _CANDIDATE),)),
  collision_primitive._primitive_narrowphase: (_CANDIDATE, (("naconmax_in", _CANDIDATE),)),
  constraint._efc_contact_init: (_CANDIDATE, ()),
  constraint._efc_contact_jac_dense: (_WORLD, ()),
  constraint._efc_contact_jac_sparse: (_CANDIDATE, ()),
  constraint._efc_contact_update: (_CANDIDATE, ()),
  constraint._friction_dof: (_WORLD, ()),
  constraint._limit_slide_hinge: (_WORLD, ()),
  forward._next_time_builder: (_WORLD, (("nworld_in", _WORLD), ("naconmax_in", _CANDIDATE))),
  forward._qfrc_smooth: (_WORLD, ()),
  island._compact_dofs_builder: (_WORLD, ()),
  passive._qfrc_passive_kernel: (_WORLD, ()),
  smooth._small_cholesky_factorize_solve_block: (_WORLD, ()),
  solver._JTDACJ_sparse: (_WORLD, ()),
  solver._cholesky_factorize_solve_blocked: (_WORLD, ()),
  solver._linesearch_iterative_kernel: (_WORLD, ()),
  solver._linesearch_jv_fused_kernel: (_WORLD, ()),
  solver._solve_done: (_WORLD, ()),
  solver._solve_init_dof: (_WORLD, ()),
  solver._solve_init_jaref_kernel: (_WORLD, ()),
  solver._update_constraint_efc: (_WORLD, ()),
  solver._update_constraint_init_qfrc_constraint_dense: (_WORLD, ()),
  solver._update_constraint_init_qfrc_constraint_sparse: (_WORLD, ()),
  solver._update_gradient_JTDAJ_dense_tiled_compact: (_WORLD, ()),
  solver._update_gradient_cholesky: (_WORLD, ()),
  solver._update_gradient_cholesky_blocked: (_WORLD, ()),
  solver._update_gradient_cholesky_blocked_skip_unchanged: (_WORLD, ()),
  solver._update_gradient_grad: (_WORLD, ()),
  solver._update_gradient_h_incremental_sparse: (_WORLD, ()),
  solver._update_gradient_init_h_sparse: (_WORLD, ()),
  solver._update_gradient_zero_grad_dot: (_WORLD, ()),
  support.mul_m_kernel: (_WORLD, ()),
}


def _kernel_contracts():
  """Resolve already materialized factory instances by exact owner identity."""
  contracts = dict(_KERNEL_DOMAINS)
  for factory, contract in _FACTORY_DOMAINS.items():
    for kernel in kernel_instances(factory):
      if kernel in contracts and contracts[kernel] != contract:
        raise ValueError("One native kernel has conflicting domain declarations")
      contracts[kernel] = contract
  return contracts


def bind_step_program(workspace, graph: wp.Graph, launches: tuple[wp.CapturedLaunch, ...]) -> None:
  """Compile exact numerical records using the prepared owner's native domains.

  The caller selects this population's records, excluding application callbacks.
  Native kernels have explicit count declarations. Ordinary Warp fills and copies
  require canonical memory-operation records and exact registered field descriptors;
  global counters are explicitly fixed. Unknown operations prevent publication.
  Generic batch adoption proves record identity, complete coverage and dependencies.
  """
  bindings = workspace.bindings
  try:
    validate_step_workspace(workspace, workspace.model, workspace.data)
    if bindings is None or bindings.updates is None:
      raise ValueError("Prepare native graph updates before adopting a step")
    if type(launches) is not tuple:
      raise TypeError("Native program launches must be an explicit immutable tuple")
    _retain_bindings(bindings, graph)
    sources = (
      (bindings.world_storage, bindings.world_storage.protected_count),
      (bindings.contact_storage, bindings.contact_storage.ready_count),
      (bindings.ccd_storage, bindings.ccd_storage.ready_count),
    )
    fixed_arrays = [workspace.data.nacon, workspace.data.ncollision, bindings.world_storage.protected_count]
    fixed_arrays.extend(workspace.arrays[spec.name] for spec in workspace._specs if spec.capacity_domain == "global_counter")
    fixed_arrays = tuple({(id(a.device), a.ptr, a.shape, a.strides, a.dtype): a for a in fixed_arrays}.values())
    launch_ids = {id(record) for record in launches}
    operations = tuple(op for op in wp.capture_get_memory_operations(graph) if id(op.launch) in launch_ids)
    memory_extents = field_ops.memory_operation_extents(graph, operations, sources, fixed_arrays=fixed_arrays)
    memory = {id(operation.launch): extent for operation, extent in zip(operations, memory_extents)}
    contracts = _kernel_contracts()
    extents, parameters, fixed = [], [], []
    for index, record in enumerate(launches):
      if id(record) in memory:
        axis, count = memory[id(record)]
        if axis is None:
          fixed.append(index)
        else:
          extents.append((index, axis, count))
        continue
      if record.kernel not in contracts:
        raise ValueError(f"Captured kernel has no native count declaration: {record.kernel.key}")
      extent, declarations = contracts[record.kernel]
      if extent is not None and record.dim[0] != sources[extent][0].capacity:
        raise ValueError("Native captured extent must match its declared population capacity")
      count, scalars = resolve_step_counts(
        bindings,
        record.kernel,
        _DOMAIN_NAMES[extent] if extent is not None else None,
        0 if extent is not None else None,
        {label: _DOMAIN_NAMES[domain] for label, domain in declarations},
      )
      if extent is not None:
        extents.append((index, 0, count))
      parameters.extend((index, parameter) for parameter in scalars)
      if extent is None and not scalars:
        fixed.append(index)
    bindings.bindings.extend(
      graph_ops.adopt_launches(bindings.updates, graph, launches, extents=extents, parameters=parameters, fixed=fixed)
    )
  except BaseException:
    _fail_recording(bindings)
    graph_ops.invalidate(graph)
    raise
