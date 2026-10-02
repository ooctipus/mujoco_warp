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
from operands. Scratch allocation and count-safe memory operations remain with
workspace preparation and step_execution.
"""

from __future__ import annotations

import warp as wp
from gpu_components import fields as field_ops
from gpu_components import graph as graph_ops

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
from mujoco_warp._src import step_execution
from mujoco_warp._src import support
from mujoco_warp._src.warp_util import kernel_instances

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


def bind_step_program(bindings: step_execution.StepBindings, graph: wp.Graph, launches: tuple[wp.CapturedLaunch, ...]) -> None:
  """Adopt exactly the native launches supplied by the population composition root.

  Capture must be finished and all updater dependencies established. The caller
  selects this population's records explicitly, excluding its application-owned
  callbacks. Repeated kernels and equal capacities do not establish ownership.
  Every supplied record must belong to the graph by identity and either have an
  explicit native contract or be a validated count-safe field operation. Unknown
  kernels and failed adoption invalidate both the graph and native preparation.
  The existing bindings ledger remains the only owner of adopted node bindings;
  the composing engine calls graph.bind after all native/application adoption.
  """
  try:
    step_execution._validate_bindings(bindings)
    if bindings.updates is None:
      raise ValueError("Prepare native graph updates before adopting a step")
    if type(launches) is not tuple:
      raise TypeError("Native program launches must be an explicit immutable tuple")
    captured = {id(record): record for record in wp.capture_get_launches(graph)}
    if any(captured.get(id(record)) is not record for record in launches):
      raise ValueError("Native program launches must be canonical records owned by this graph")
    if len({id(record) for record in launches}) != len(launches):
      raise ValueError("Native program launches must be distinct records")
    step_execution._retain_bindings(bindings, graph)
    sources = (
      (bindings.world_storage, bindings.world_storage.protected_count),
      (bindings.contact_storage, bindings.contact_storage.ready_count),
      (bindings.ccd_storage, bindings.ccd_storage.ready_count),
    )
    contracts = _kernel_contracts()
    for record in launches:
      if field_ops.validate_captured_operation(record, sources):
        continue
      if record.kernel not in contracts:
        raise ValueError(f"Captured kernel has no native count declaration: {record.kernel.key}")
      extent, declarations = contracts[record.kernel]
      if extent is not None and record.dim[0] != sources[extent][0].capacity:
        raise ValueError("Native captured extent must match its declared population capacity")
      count, parameters = step_execution.resolve_step_counts(
        bindings,
        record.kernel,
        _DOMAIN_NAMES[extent] if extent is not None else None,
        0 if extent is not None else None,
        {label: _DOMAIN_NAMES[domain] for label, domain in declarations},
      )
      binding = graph_ops.adopt_launch(
        bindings.updates,
        graph,
        record,
        extent_axis=0 if extent is not None else None,
        extent_source=count,
        parameters=parameters,
      )
      bindings.bindings.append(binding)
  except BaseException:
    step_execution._fail_recording(bindings)
    graph_ops.invalidate(graph)
    raise
