# Copyright 2026 The Newton Developers
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

"""Per-world CTA fusion of the forward dynamics and integration for small sleeping models.

Four kernels, one 128-thread CTA per world, replace the launch sequence of ``step()`` except the
constraint solver for models that satisfy :func:`fused_world`:

* ``forward_a``: ``sleep.wake`` + ``sleep.update_sleep_trees``, kinematics, ``com_pos``, ``crb`` and
  the sparse inertia ``M``, joint transmission, ``com_vel``, passive forces (springs, dampers,
  gravity compensation), RNE bias forces and the position/velocity servo actuator forces.
* ``forward_m``: ``constraint.make_constraint`` (JOINT equalities, dof friction, slide/hinge limits
  and pyramidal contact rows from the world's contact list), ``sleep.wake_equality``,
  ``sleep.update_sleep`` and the ``island`` bitset flood fill. Two small grouping launches (a
  counting sort of the live contacts by world) precede it because externally supplied contacts carry
  a global id order.
* ``forward_b``: ``qfrc_smooth`` (with the sleeping-tree freeze), ``xfrc_accumulate``, the per-tree
  factor/solve for ``qacc_smooth`` (``qLD``/``qLDiagInv``) and the active-DOF compaction maps that
  the compact constraint solver consumes.
* ``forward_c`` (after ``solver.solve``): the implicitfast factor ``M - h*D`` and solve with rhs
  ``efc.Ma``, velocity/position/time advance with the overflow flags, the warmstart copy,
  ``sleep.sleep`` (countdown, island can-sleep, cycles with zeroed ``qvel``/``qacc``), the
  post-sleep ``fwd_velocity`` refresh when the step finalizes and ``sleep.update_sleep``.

Every canonical ``Data`` field written by the replaced kernels is published so that downstream
consumers see identical state. Integer and sleep/island state is reproduced exactly; floating point
differs from the stock launches only through summation and factorization order (scalar instead of
tile Cholesky for the dense inertia blocks: fp32 rounding, a few ulps in ``qLD``/``qacc_smooth``).
Row order within a constraint family and the ``body_awake_ind``/``dof_awake_ind`` order are
deterministic given the contact list order (stock: atomic allocation order); the contact list order
itself follows the grouping atomics.

Per-body and per-dof state lives in shared memory. Tree traversals use the depth-first body
numbering of MuJoCo (a subtree is a contiguous id range), so backward accumulations are gather sums
without atomics, and forward passes are level-synchronous with all bodies of a depth processed in
parallel. Bodies (with their single joint), dofs, actuators and trees are owned by one thread each;
the block dimension equals the dof capacity.

Warp pitfalls handled here: tile element assignment embeds a block barrier, so shared stores in
thread-divergent code use single-line native snippets; multi-line native snippets live at module
scope.
"""

import os

import numpy as np
import warp as wp

from mujoco_warp._src import constraint
from mujoco_warp._src import math
from mujoco_warp._src import types
from mujoco_warp._src import util_misc
from mujoco_warp._src.types import Q_LD_BLOCK_COMPACT
from mujoco_warp._src.types import Q_LD_BLOCK_SPARSE
from mujoco_warp._src.types import BiasType
from mujoco_warp._src.types import ConeType
from mujoco_warp._src.types import ConstraintType
from mujoco_warp._src.types import ContactType
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import DisableBit
from mujoco_warp._src.types import DynType
from mujoco_warp._src.types import EnableBit
from mujoco_warp._src.types import EqType
from mujoco_warp._src.types import GainType
from mujoco_warp._src.types import IntegratorType
from mujoco_warp._src.types import JointType
from mujoco_warp._src.types import Model
from mujoco_warp._src.types import OverflowType
from mujoco_warp._src.types import SleepPolicy
from mujoco_warp._src.types import SleepState
from mujoco_warp._src.types import SolverType
from mujoco_warp._src.types import TrnType
from mujoco_warp._src.types import vec5
from mujoco_warp._src.types import vec10
from mujoco_warp._src.types import vec11
from mujoco_warp._src.warp_util import cache_kernel
from mujoco_warp._src.warp_util import event_scope

wp.set_module_options({"enable_backward": False})

# Compile-time capacities of the per-world CTA. The block dimension equals the dof capacity so
# that every dof is owned by exactly one thread.
NV_CAP = 128
NBODY_CAP = 64
NU_CAP = 128
NTREE_CAP = 32
# dofs per tree handled by the dense scalar Cholesky; trees above six dofs use a wider scratch slot
NVTREE_SMALL = 6
NVTREE_CAP = 16
NBIG_CAP = 2

# tree_asleep value for a fully awake tree (see sleep.py)
_K_AWAKE_VAL = -(1 + types.MJ_MINAWAKE)

# Host switch (e.g. for A/B comparisons); MJWARP_FUSED_WORLD=0 disables the fused path.
enabled = os.environ.get("MJWARP_FUSED_WORLD", "1") != "0"


@wp.func_native(snippet="WP_TILE_SYNC();")
def _sync():
  pass


# Warp-level ballot/popcount for deterministic in-block compaction (all 32 lanes must participate).
# Multi-line snippets must live at module scope: Warp dedents the function source before parsing.
@wp.func_native(
  snippet="""
#if defined(__CUDA_ARCH__)
return __ballot_sync(0xffffffffu, pred != 0);
#else
return (unsigned)0;
#endif
"""
)
def _ballot(pred: int) -> wp.uint32: ...


@wp.func_native(
  snippet="""
#if defined(__CUDA_ARCH__)
return __popc(mask);
#else
return 0;
#endif
"""
)
def _popc(mask: wp.uint32) -> int: ...


@wp.func_native(
  snippet="""
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
return __reduce_or_sync(0xffffffffu, value);
#elif defined(__CUDA_ARCH__)
for (int offset = 16; offset > 0; offset >>= 1) value |= __shfl_xor_sync(0xffffffffu, value, offset);
return value;
#else
return value;
#endif
"""
)
def _warp_or(value: wp.uint32) -> wp.uint32: ...


@wp.func_native(
  snippet="""
#if defined(__CUDA_ARCH__)
for (int offset = 1; offset < 32; offset <<= 1) {
  int other = __shfl_up_sync(0xffffffffu, value, offset);
  if ((threadIdx.x & 31) >= offset) value += other;
}
return value;
#else
return value;
#endif
"""
)
def _warp_scan_inclusive(value: int) -> int: ...


@wp.func_native(
  snippet="""
#if defined(__CUDA_ARCH__)
return (int)threadIdx.x;
#else
return 0;
#endif
"""
)
def _thread_index() -> int: ...


# module-level shared-store snippets for the tiles shared between the fused kernels and their
# helpers
@wp.func_native(snippet="values.data(wp::tile_coord(index)) = value;")
def _st_scan_i(values: wp.tile[int, 8], index: int, value: int): ...


@wp.func_native(snippet="values.data(wp::tile_coord(index)) = value;")
def _st_misc16_i(values: wp.tile[int, 16], index: int, value: int): ...


@wp.func_native(snippet="values.data(wp::tile_coord(index)) = value;")
def _st_tree32_i(values: wp.tile[int, NTREE_CAP], index: int, value: int): ...


@wp.func_native(snippet="values.data(wp::tile_coord(index)) = value;")
def _st_body64_i(values: wp.tile[int, NBODY_CAP], index: int, value: int): ...


@wp.func_native(snippet="values.data(wp::tile_coord(index)) = value;")
def _st_tree32_u(values: wp.tile[wp.uint32, NTREE_CAP], index: int, value: wp.uint32): ...


@wp.func_native(snippet="atomicOr((unsigned int*)&values.data(wp::tile_coord(index)), value);")
def _or_tree32_u(values: wp.tile[wp.uint32, NTREE_CAP], index: int, value: wp.uint32): ...


@wp.func_native(snippet="values.data(wp::tile_coord(index)) = value;")
def _st_chunk_i(values: wp.tile[int, NV_CAP], index: int, value: int): ...


@wp.func
def _block_scan(scratch: wp.tile[int, 8], value: int, lane: int, warp: int) -> int:
  """Exclusive prefix of ``value`` over the 128-thread block; the total is left in ``scratch[4]``.

  Contains block barriers: every thread must call it (pass 0 for idle threads).
  """
  inclusive = _warp_scan_inclusive(value)
  if lane == 31:
    _st_scan_i(scratch, warp, inclusive)
  _sync()
  exclusive = inclusive - value
  for w in range(warp):
    exclusive += scratch[w]
  if lane == 0 and warp == 0:
    _st_scan_i(scratch, 4, scratch[0] + scratch[1] + scratch[2] + scratch[3])
  _sync()
  return exclusive


@wp.func
def _sleep_cycle_sh(tree_asleep_sh: wp.tile[int, NTREE_CAP], ntree: int, treeid: int) -> int:
  """Smallest tree id of the sleep cycle containing ``treeid`` (sleep._sleep_cycle on shared)."""
  if treeid < 0 or treeid >= ntree:
    return -1
  smallest = int(treeid)
  current = int(treeid)
  for _step in range(ntree + 1):
    next_tree = tree_asleep_sh[current]
    if next_tree < 0 or next_tree >= ntree:
      return -1
    if next_tree < smallest:
      smallest = next_tree
    current = next_tree
    if current == treeid:
      break
  return smallest


@wp.func
def _wake_tree_sh(tree_asleep_sh: wp.tile[int, NTREE_CAP], ntree: int, treeid: int, wakeval: int):
  """Wake ``treeid`` and its sleep cycle (sleep._wake_tree on shared state)."""
  if treeid < 0 or treeid >= ntree:
    return
  asleep_val = tree_asleep_sh[treeid]
  if asleep_val < 0:
    if wakeval < asleep_val:
      _st_tree32_i(tree_asleep_sh, treeid, wakeval)
    return
  current = int(treeid)
  for _step in range(ntree + 1):
    next_tree = tree_asleep_sh[current]
    if next_tree < 0 or next_tree >= ntree:
      break
    _st_tree32_i(tree_asleep_sh, current, wakeval)
    current = next_tree
    if current == treeid:
      break


@wp.func
def _publish_sleep_state(
  tree_asleep_sh: wp.tile[int, NTREE_CAP],
  body_awake_sh: wp.tile[int, NBODY_CAP],
  misc_sh: wp.tile[int, 16],
  nbody: int,
  nv: int,
  ntree: int,
  body_rootid: wp.array[int],
  body_mocapid: wp.array[int],
  body_treeid: wp.array[int],
  dof_bodyid: wp.array[int],
  worldid: int,
  tid: int,
  tree_asleep_out: wp.array2d[int],
  tree_awake_out: wp.array2d[int],
  ntree_awake_out: wp.array[int],
  body_awake_out: wp.array2d[int],
  body_awake_ind_out: wp.array2d[int],
  nbody_awake_out: wp.array[int],
  dof_awake_ind_out: wp.array2d[int],
  nv_awake_out: wp.array[int],
):
  """sleep.update_sleep from the shared ``tree_asleep``.

  ``body_awake_ind``/``dof_awake_ind`` are compacted in body/dof order (stock: atomic order); their
  sole external consumer copies whole arrays. Contains block barriers; every thread must call it.
  """
  lane = tid & 31
  warp = tid >> 5
  lanemask = (wp.uint32(1) << wp.uint32(lane)) - wp.uint32(1)

  # trees (ntree <= 32: all in the first warp)
  awake_flag = int(0)
  if tid < ntree:
    asleep = tree_asleep_sh[tid]
    tree_asleep_out[worldid, tid] = asleep
    if asleep < 0:
      awake_flag = 1
    tree_awake_out[worldid, tid] = awake_flag
  awake_mask = _ballot(awake_flag)
  if tid == 0:
    ntree_awake_out[worldid] = _popc(awake_mask)

  # bodies: STATIC for world-attached bodies (AWAKE when descended from a mocap root)
  body_flag = int(0)
  if tid < nbody:
    tree = body_treeid[tid]
    state = int(SleepState.STATIC)
    if tree < 0:
      if body_mocapid[body_rootid[tid]] >= 0:
        state = int(SleepState.AWAKE)
    elif tree_asleep_sh[tree] < 0:
      state = int(SleepState.AWAKE)
    else:
      state = int(SleepState.ASLEEP)
    _st_body64_i(body_awake_sh, tid, state)
    body_awake_out[worldid, tid] = state
    if state != SleepState.ASLEEP:
      body_flag = 1
  body_mask = _ballot(body_flag)
  if lane == 0:
    _st_misc16_i(misc_sh, warp, _popc(body_mask))
  _sync()
  if body_flag != 0:
    offset = int(0)
    for w in range(warp):
      offset += misc_sh[w]
    body_awake_ind_out[worldid, offset + _popc(body_mask & lanemask)] = tid
  if tid == 0:
    nbody_awake_out[worldid] = misc_sh[0] + misc_sh[1] + misc_sh[2] + misc_sh[3]

  # dofs of awake, non-static bodies
  dof_flag = int(0)
  if tid < nv:
    body = dof_bodyid[tid]
    if body_treeid[body] >= 0 and body_awake_sh[body] == SleepState.AWAKE:
      dof_flag = 1
  dof_mask = _ballot(dof_flag)
  if lane == 0:
    _st_misc16_i(misc_sh, 4 + warp, _popc(dof_mask))
  _sync()
  if dof_flag != 0:
    offset = int(0)
    for w in range(warp):
      offset += misc_sh[4 + w]
    dof_awake_ind_out[worldid, offset + _popc(dof_mask & lanemask)] = tid
  if tid == 0:
    nv_awake_out[worldid] = misc_sh[4] + misc_sh[5] + misc_sh[6] + misc_sh[7]


@wp.func
def _mark_tree_edge(edge_sh: wp.tile[wp.uint32, NTREE_CAP], tree0: int, tree1: int):
  """Edge rule of island._tree_edges: self-edge for one-tree rows, symmetric bits otherwise."""
  if tree0 < 0 and tree1 >= 0:
    tree0 = tree1
    tree1 = -1
  if tree0 >= 0:
    if tree1 < 0 or tree0 == tree1:
      _or_tree32_u(edge_sh, tree0, wp.uint32(1) << wp.uint32(tree0))
    else:
      t1 = wp.min(tree0, tree1)
      t2 = wp.max(tree0, tree1)
      _or_tree32_u(edge_sh, t1, wp.uint32(1) << wp.uint32(t2))
      _or_tree32_u(edge_sh, t2, wp.uint32(1) << wp.uint32(t1))


@wp.func
def _lowest_set_bit(mask: wp.uint32) -> int:
  index = int(0)
  if (mask & wp.uint32(0xFFFF)) == 0:
    mask >>= wp.uint32(16)
    index += 16
  if (mask & wp.uint32(0xFF)) == 0:
    mask >>= wp.uint32(8)
    index += 8
  if (mask & wp.uint32(0xF)) == 0:
    mask >>= wp.uint32(4)
    index += 4
  if (mask & wp.uint32(0x3)) == 0:
    mask >>= wp.uint32(2)
    index += 2
  if (mask & wp.uint32(0x1)) == 0:
    index += 1
  return index


def _dfs_contiguous(body_parentid: np.ndarray) -> bool:
  """Whether every subtree occupies a contiguous id range (MuJoCo's depth-first numbering)."""
  nbody = body_parentid.shape[0]
  end = np.zeros(nbody, dtype=int)
  for b in range(nbody - 1, -1, -1):
    end[b] = b + 1
    for c in range(b + 1, nbody):
      if body_parentid[c] == b:
        end[b] = max(end[b], end[c])
  for b in range(nbody):
    for c in range(b + 1, end[b]):
      # every body in the range must descend from b
      p = c
      while p > b:
        p = body_parentid[p]
      if p != b:
        return False
  return True


def static_eligible(mjm, m: Model) -> bool:
  """Structural (model-only) part of the fused-world predicate, evaluated on host data in put_model.

  Args:
    mjm: The MuJoCo model (host arrays).
    m: The partially built MJWarp model; its derived scalars and the host ``qLD_block_adr`` layout
      are read before they are converted to device arrays.
  """
  if mjm.nv == 0 or mjm.ntree == 0 or mjm.nbody < 2:
    return False
  if mjm.nv > NV_CAP or mjm.nbody > NBODY_CAP or mjm.nu > NU_CAP or mjm.ntree > NTREE_CAP:
    return False
  if mjm.ntendon or mjm.nflex or mjm.na or m.nhistory or m.nacttrnbody:
    return False
  if m.has_fluid or m.flg_adhesion or not m.is_sparse:
    return False

  jnt_type = np.asarray(mjm.jnt_type)
  if not np.isin(jnt_type, (JointType.FREE, JointType.HINGE, JointType.SLIDE)).all():
    return False
  if mjm.nbody > 1 and np.asarray(mjm.body_jntnum).max() > 1:
    return False
  if mjm.neq and not (np.asarray(mjm.eq_type) == EqType.JOINT).all():
    return False

  if mjm.nu:
    if not np.isin(mjm.actuator_trntype, (TrnType.JOINT, TrnType.JOINTINPARENT)).all():
      return False
    if not np.isin(jnt_type[np.asarray(mjm.actuator_trnid)[:, 0]], (JointType.HINGE, JointType.SLIDE)).all():
      return False
    if not (np.asarray(mjm.actuator_dyntype) == DynType.NONE).all():
      return False
    if not np.isin(mjm.actuator_gaintype, (GainType.FIXED, GainType.AFFINE)).all():
      return False
    if not np.isin(mjm.actuator_biastype, (BiasType.NONE, BiasType.AFFINE)).all():
      return False
    if np.asarray(mjm.actuator_actearly).any() or mjm.nJmom < mjm.nu:
      return False

  tree_dofnum = np.asarray(mjm.tree_dofnum)
  if tree_dofnum.max() > NVTREE_CAP or (tree_dofnum > NVTREE_SMALL).sum() > NBIG_CAP:
    return False
  if (np.asarray(m.qLD_block_adr) == Q_LD_BLOCK_SPARSE).any():
    return False

  return _dfs_contiguous(np.asarray(mjm.body_parentid))


def fused_world(m: Model, d: Data) -> bool:
  """Return whether the fused per-world forward kernels apply to ``(m, d)``.

  The structural part comes from ``m.fused_world_static`` (host data in put_model); option flags and
  runtime toggles (such as ``sensor_rne_postconstraint``, which Newton flips at runtime) are
  re-evaluated on every call. Only host state is read, so the predicate is safe under graph capture.
  """
  if not enabled or not getattr(m, "fused_world_static", False):
    return False
  opt = m.opt
  if opt.solver != SolverType.NEWTON or opt.cone != ConeType.PYRAMIDAL or opt.integrator != IntegratorType.IMPLICITFAST:
    return False
  if opt.run_collision_detection:
    return False
  if not (opt.enableflags & EnableBit.SLEEP) or (opt.disableflags & DisableBit.ISLAND):
    return False
  if m.sensor_rne_postconstraint:
    return False
  if any(callback is not None for callback in vars(m.callback).values()):
    return False
  if d.nworld == 0 or not d.qpos.device.is_cuda:
    return False
  if d.qLD.shape[1] != m.qLD_block_total:
    return False
  return True


@cache_kernel
def _forward_a_kernel(NB: int, NV: int, NU: int, NT: int):
  BLOCK = NV

  # tile element stores through native snippets: the builtin element assignment embeds a block
  # barrier, which is not allowed inside thread-divergent phases
  @wp.func_native(snippet="values.data(wp::tile_coord(index)) = value;")
  def _st_body_v3(values: wp.tile[wp.vec3, NB], index: int, value: wp.vec3): ...

  @wp.func_native(snippet="values.data(wp::tile_coord(index)) = value;")
  def _st_body_q(values: wp.tile[wp.quat, NB], index: int, value: wp.quat): ...

  @wp.func_native(snippet="values.data(wp::tile_coord(index)) = value;")
  def _st_body_v10(values: wp.tile[vec10, NB], index: int, value: vec10): ...

  @wp.func_native(snippet="values.data(wp::tile_coord(index)) = value;")
  def _st_body_sv(values: wp.tile[wp.spatial_vector, NB], index: int, value: wp.spatial_vector): ...

  @wp.func_native(snippet="values.data(wp::tile_coord(index)) = value;")
  def _st_body_i(values: wp.tile[int, NB], index: int, value: int): ...

  @wp.func_native(snippet="values.data(wp::tile_coord(index)) = value;")
  def _st_dof_sv(values: wp.tile[wp.spatial_vector, NV], index: int, value: wp.spatial_vector): ...

  @wp.func_native(snippet="values.data(wp::tile_coord(index)) = value;")
  def _st_dof_f(values: wp.tile[float, NV], index: int, value: float): ...

  @wp.func_native(snippet="values.data(wp::tile_coord(index)) = value;")
  def _st_dof_i(values: wp.tile[int, NV], index: int, value: int): ...

  @wp.func_native(snippet="values.data(wp::tile_coord(index)) = value;")
  def _st_act_f(values: wp.tile[float, NU], index: int, value: float): ...

  @wp.func_native(snippet="values.data(wp::tile_coord(index)) = value;")
  def _st_act_i(values: wp.tile[int, NU], index: int, value: int): ...

  @wp.func_native(snippet="values.data(wp::tile_coord(index)) = value;")
  def _st_tree_i(values: wp.tile[int, NT], index: int, value: int): ...

  @wp.func_native(snippet="values.data(wp::tile_coord(index)) = value;")
  def _st_misc_i(values: wp.tile[int, 16], index: int, value: int): ...

  @wp.kernel(module="unique", enable_backward=False, grid_stride=False, launch_bounds=(BLOCK,))
  def kernel(
    # Model:
    nbody: int,
    nv: int,
    nu: int,
    ntree: int,
    ngeom: int,
    nsite: int,
    ngravcomp: int,
    opt_disableflags: int,
    opt_gravity: wp.array[wp.vec3],
    qpos0: wp.array2d[float],
    qpos_spring: wp.array2d[float],
    body_parentid: wp.array[int],
    body_rootid: wp.array[int],
    body_weldid: wp.array[int],
    body_mocapid: wp.array[int],
    body_jntnum: wp.array[int],
    body_jntadr: wp.array[int],
    body_dofnum: wp.array[int],
    body_dofadr: wp.array[int],
    body_treeid: wp.array[int],
    body_pos: wp.array2d[wp.vec3],
    body_quat: wp.array2d[wp.quat],
    body_ipos: wp.array2d[wp.vec3],
    body_iquat: wp.array2d[wp.quat],
    body_mass: wp.array2d[float],
    body_subtreemass: wp.array2d[float],
    body_inertia: wp.array2d[wp.vec3],
    body_gravcomp: wp.array2d[float],
    body_tree_offsets: wp.array[int],
    gravcomp_bodyid: wp.array[int],
    jnt_type: wp.array[int],
    jnt_qposadr: wp.array[int],
    jnt_dofadr: wp.array[int],
    jnt_pos: wp.array2d[wp.vec3],
    jnt_axis: wp.array2d[wp.vec3],
    jnt_stiffness: wp.array2d[float],
    jnt_stiffnesspoly: wp.array2d[wp.vec2],
    jnt_actgravcomp: wp.array[int],
    jnt_actfrclimited: wp.array[bool],
    jnt_actfrcrange: wp.array2d[wp.vec2],
    dof_bodyid: wp.array[int],
    dof_parentid: wp.array[int],
    dof_jntid: wp.array[int],
    dof_armature: wp.array2d[float],
    dof_damping: wp.array2d[float],
    dof_dampingpoly: wp.array2d[wp.vec2],
    tree_sleep_policy: wp.array[int],
    M_rownnz: wp.array[int],
    M_rowadr: wp.array[int],
    geom_bodyid: wp.array[int],
    geom_pos: wp.array2d[wp.vec3],
    geom_quat: wp.array2d[wp.quat],
    site_bodyid: wp.array[int],
    site_pos: wp.array2d[wp.vec3],
    site_quat: wp.array2d[wp.quat],
    actuator_trnid: wp.array[wp.vec2i],
    actuator_gear: wp.array2d[wp.spatial_vector],
    actuator_gaintype: wp.array[int],
    actuator_biastype: wp.array[int],
    actuator_gainprm: wp.array2d[vec10],
    actuator_biasprm: wp.array2d[vec10],
    actuator_ctrllimited: wp.array[bool],
    actuator_ctrlrange: wp.array2d[wp.vec2],
    actuator_forcelimited: wp.array[bool],
    actuator_forcerange: wp.array2d[wp.vec2],
    # Data in:
    qpos_in: wp.array2d[float],
    qvel_in: wp.array2d[float],
    qfrc_applied_in: wp.array2d[float],
    xfrc_applied_in: wp.array2d[wp.spatial_vector],
    mocap_pos_in: wp.array2d[wp.vec3],
    mocap_quat_in: wp.array2d[wp.quat],
    ctrl_in: wp.array2d[float],
    # Data out:
    tree_asleep_out: wp.array2d[int],
    tree_awake_out: wp.array2d[int],
    ntree_awake_out: wp.array[int],
    body_awake_out: wp.array2d[int],
    body_awake_ind_out: wp.array2d[int],
    nbody_awake_out: wp.array[int],
    dof_awake_ind_out: wp.array2d[int],
    nv_awake_out: wp.array[int],
    xpos_out: wp.array2d[wp.vec3],
    xquat_out: wp.array2d[wp.quat],
    xmat_out: wp.array2d[wp.mat33],
    xipos_out: wp.array2d[wp.vec3],
    ximat_out: wp.array2d[wp.mat33],
    xanchor_out: wp.array2d[wp.vec3],
    xaxis_out: wp.array2d[wp.vec3],
    geom_xpos_out: wp.array2d[wp.vec3],
    geom_xmat_out: wp.array2d[wp.mat33],
    site_xpos_out: wp.array2d[wp.vec3],
    site_xmat_out: wp.array2d[wp.mat33],
    subtree_com_out: wp.array2d[wp.vec3],
    cinert_out: wp.array2d[vec10],
    cdof_out: wp.array2d[wp.spatial_vector],
    crb_out: wp.array2d[vec10],
    M_out: wp.array2d[float],
    actuator_length_out: wp.array2d[float],
    moment_rownnz_out: wp.array2d[int],
    moment_rowadr_out: wp.array2d[int],
    moment_colind_out: wp.array2d[int],
    actuator_moment_out: wp.array2d[float],
    actuator_velocity_out: wp.array2d[float],
    cvel_out: wp.array2d[wp.spatial_vector],
    cdof_dot_out: wp.array2d[wp.spatial_vector],
    qfrc_spring_out: wp.array2d[float],
    qfrc_damper_out: wp.array2d[float],
    qfrc_gravcomp_out: wp.array2d[float],
    qfrc_adhesion_out: wp.array2d[float],
    qfrc_passive_out: wp.array2d[float],
    cacc_out: wp.array2d[wp.spatial_vector],
    cfrc_int_out: wp.array2d[wp.spatial_vector],
    qfrc_bias_out: wp.array2d[float],
    actuator_force_out: wp.array2d[float],
    qfrc_actuator_out: wp.array2d[float],
  ):
    worldid, tid = wp.tid()

    # shared arenas (per body / per dof / per actuator / per tree)
    xpos_sh = wp.tile_empty(shape=(NB,), dtype=wp.vec3, storage="shared")
    xquat_sh = wp.tile_empty(shape=(NB,), dtype=wp.quat, storage="shared")
    xipos_sh = wp.tile_empty(shape=(NB,), dtype=wp.vec3, storage="shared")
    com_sh = wp.tile_empty(shape=(NB,), dtype=wp.vec3, storage="shared")
    gforce_sh = wp.tile_empty(shape=(NB,), dtype=wp.vec3, storage="shared")
    cinert_sh = wp.tile_empty(shape=(NB,), dtype=vec10, storage="shared")
    crb_sh = wp.tile_empty(shape=(NB,), dtype=vec10, storage="shared")
    cvel_sh = wp.tile_empty(shape=(NB,), dtype=wp.spatial_vector, storage="shared")
    cacc_sh = wp.tile_empty(shape=(NB,), dtype=wp.spatial_vector, storage="shared")
    parent_sh = wp.tile_empty(shape=(NB,), dtype=int, storage="shared")
    send_sh = wp.tile_empty(shape=(NB,), dtype=int, storage="shared")
    body_awake_sh = wp.tile_empty(shape=(NB,), dtype=int, storage="shared")
    gc_body_sh = wp.tile_empty(shape=(NB,), dtype=int, storage="shared")
    cdof_sh = wp.tile_empty(shape=(NV,), dtype=wp.spatial_vector, storage="shared")
    qvel_sh = wp.tile_empty(shape=(NV,), dtype=float, storage="shared")
    dof_parent_sh = wp.tile_empty(shape=(NV,), dtype=int, storage="shared")
    act_force_sh = wp.tile_empty(shape=(NU,), dtype=float, storage="shared")
    act_dof_sh = wp.tile_empty(shape=(NU,), dtype=int, storage="shared")
    tree_asleep_sh = wp.tile_empty(shape=(NT,), dtype=int, storage="shared")
    tree_awake_sh = wp.tile_empty(shape=(NT,), dtype=int, storage="shared")
    misc_sh = wp.tile_empty(shape=(16,), dtype=int, storage="shared")

    lane = tid & 31
    warp = tid >> 5
    lanemask = (wp.uint32(1) << wp.uint32(lane)) - wp.uint32(1)

    is_body = tid < nbody
    is_dof = tid < nv
    is_tree = tid < ntree
    nlevel = body_tree_offsets.shape[0] - 1

    gravity_enabled = (opt_disableflags & DisableBit.GRAVITY) == 0
    dsbl_spring = (opt_disableflags & DisableBit.SPRING) != 0
    dsbl_damper = (opt_disableflags & DisableBit.DAMPER) != 0
    gravity = opt_gravity[worldid % opt_gravity.shape[0]]

    # ------------------------------------------------------------- P0: prefetch (one latency round)
    # body-owned parameters (bodies have at most one joint, see fused_world())
    b_parent = int(0)
    b_tree = int(-1)
    b_root = int(0)
    b_mocap = int(-1)
    b_jntadr = int(-1)
    b_jntnum = int(0)
    b_dofadr = int(-1)
    b_dofnum = int(0)
    b_mass = float(0.0)
    b_gravcomp = float(0.0)
    b_xfrc_nz = int(0)
    j_type = int(-1)
    j_qadr = int(0)
    j_axis = wp.vec3(0.0)
    j_pos = wp.vec3(0.0)
    j_q = float(0.0)
    j_q0 = float(0.0)
    f_pos = wp.vec3(0.0)
    f_quat = wp.quat(1.0, 0.0, 0.0, 0.0)
    b_pos = wp.vec3(0.0)
    b_quat = wp.quat(1.0, 0.0, 0.0, 0.0)
    if is_body:
      b_parent = body_parentid[tid]
      b_tree = body_treeid[tid]
      b_root = body_rootid[tid]
      b_mocap = body_mocapid[tid]
      b_jntadr = body_jntadr[tid]
      b_jntnum = body_jntnum[tid]
      b_dofadr = body_dofadr[tid]
      b_dofnum = body_dofnum[tid]
      b_mass = body_mass[worldid % body_mass.shape[0], tid]
      b_gravcomp = body_gravcomp[worldid % body_gravcomp.shape[0], tid]
      b_pos = body_pos[worldid % body_pos.shape[0], tid]
      b_quat = body_quat[worldid % body_quat.shape[0], tid]
      if not (xfrc_applied_in[worldid, tid] == wp.spatial_vector()):
        b_xfrc_nz = 1
      _st_body_i(parent_sh, tid, b_parent)
      if b_jntnum == 1:
        j_type = jnt_type[b_jntadr]
        j_qadr = jnt_qposadr[b_jntadr]
        j_axis = jnt_axis[worldid % jnt_axis.shape[0], b_jntadr]
        j_pos = jnt_pos[worldid % jnt_pos.shape[0], b_jntadr]
        if j_type == JointType.FREE:
          qpos = qpos_in[worldid]
          f_pos = wp.vec3(qpos[j_qadr], qpos[j_qadr + 1], qpos[j_qadr + 2])
          f_quat = wp.quat(qpos[j_qadr + 3], qpos[j_qadr + 4], qpos[j_qadr + 5], qpos[j_qadr + 6])
        else:
          j_q = qpos_in[worldid, j_qadr]
          j_q0 = qpos0[worldid % qpos0.shape[0], j_qadr]
      if tid == 0:
        # the world body keeps whatever pose Data carries (kinematics never writes body 0)
        _st_body_v3(xpos_sh, 0, xpos_out[worldid, 0])
        _st_body_q(xquat_sh, 0, xquat_out[worldid, 0])
    # dof-owned parameters
    d_body = int(0)
    d_tree = int(-1)
    d_jnt = int(0)
    d_qvel = float(0.0)
    d_qfrc_nz = int(0)
    if is_dof:
      d_body = dof_bodyid[tid]
      d_tree = body_treeid[d_body]
      d_jnt = dof_jntid[tid]
      d_qvel = qvel_in[worldid, tid]
      if qfrc_applied_in[worldid, tid] != 0.0:
        d_qfrc_nz = 1
      _st_dof_f(qvel_sh, tid, d_qvel)
      _st_dof_i(dof_parent_sh, tid, dof_parentid[tid])
    if is_tree:
      _st_tree_i(tree_asleep_sh, tid, tree_asleep_out[worldid, tid])
      _st_tree_i(tree_awake_sh, tid, tree_awake_out[worldid, tid])
    if tid < ngravcomp:
      _st_body_i(gc_body_sh, tid, gravcomp_bodyid[tid])

    # per-tree wake masks (ntree <= 32): applied Cartesian force, applied generalized force,
    # velocity
    xfrc_bits = wp.uint32(0)
    if is_body and b_xfrc_nz != 0 and b_tree >= 0:
      xfrc_bits = wp.uint32(1) << wp.uint32(b_tree)
    qfrc_bits = wp.uint32(0)
    qvel_bits = wp.uint32(0)
    if is_dof and d_tree >= 0:
      if d_qfrc_nz != 0:
        qfrc_bits = wp.uint32(1) << wp.uint32(d_tree)
      if d_qvel != 0.0:
        qvel_bits = wp.uint32(1) << wp.uint32(d_tree)
    xfrc_bits = _warp_or(xfrc_bits)
    qfrc_bits = _warp_or(qfrc_bits)
    qvel_bits = _warp_or(qvel_bits)
    if lane == 0:
      _st_misc_i(misc_sh, warp, int(xfrc_bits))
      _st_misc_i(misc_sh, 4 + warp, int(qfrc_bits))
      _st_misc_i(misc_sh, 8 + warp, int(qvel_bits))
    _sync()

    # depth-first numbering: the subtree of tid is [tid, subtree_end); body level from the parent
    # walk
    subtree_end = int(0)
    b_level = int(0)
    if is_body:
      subtree_end = tid + 1
      while subtree_end < nbody:
        if parent_sh[subtree_end] < tid:
          break
        subtree_end += 1
      _st_body_i(send_sh, tid, subtree_end)
      p = int(tid)
      while p != 0:
        p = parent_sh[p]
        b_level += 1

    # ---------------------------------------------------------------- P1: sleep.wake
    if is_tree:
      if tree_asleep_sh[tid] >= 0:
        wake_bits = wp.uint32(0)
        for w in range(4):
          wake_bits |= wp.uint32(misc_sh[w]) | wp.uint32(misc_sh[4 + w]) | wp.uint32(misc_sh[8 + w])
        wake = int(0)
        if tree_awake_sh[tid] == 1:
          wake = 1
        elif tree_sleep_policy[tid] == SleepPolicy.AUTO_NEVER:
          wake = 1
        elif ((wake_bits >> wp.uint32(tid)) & wp.uint32(1)) != wp.uint32(0):
          wake = 1
        if wake != 0:
          # wake the whole sleep cycle (same benign cross-thread race as sleep._wake_tree)
          current = int(tid)
          for _step in range(ntree + 1):
            next_tree = tree_asleep_sh[current]
            if next_tree < 0 or next_tree >= ntree:
              break
            _st_tree_i(tree_asleep_sh, current, _K_AWAKE_VAL)
            current = next_tree
            if current == tid:
              break
    _sync()

    # ---------------------------------------------------------------- P1b: update_sleep (trees)
    awake_flag = int(0)
    if is_tree:
      asleep = tree_asleep_sh[tid]
      tree_asleep_out[worldid, tid] = asleep
      if asleep < 0:
        awake_flag = 1
      _st_tree_i(tree_awake_sh, tid, awake_flag)
      tree_awake_out[worldid, tid] = awake_flag
    awake_mask = _ballot(awake_flag)
    if tid == 0:
      # trees fit in the first warp (ntree <= 32)
      ntree_awake_out[worldid] = _popc(awake_mask)
    _sync()

    # ---------------------------------------------------------------- P1c: update_sleep (bodies)
    body_flag = int(0)
    if is_body:
      state = int(SleepState.STATIC)
      if b_tree < 0:
        if body_mocapid[b_root] >= 0:
          state = int(SleepState.AWAKE)
      elif tree_awake_sh[b_tree] == 1:
        state = int(SleepState.AWAKE)
      else:
        state = int(SleepState.ASLEEP)
      _st_body_i(body_awake_sh, tid, state)
      body_awake_out[worldid, tid] = state
      if state != SleepState.ASLEEP:
        body_flag = 1
    body_mask = _ballot(body_flag)
    if lane == 0:
      _st_misc_i(misc_sh, warp, _popc(body_mask))
    _sync()

    # deterministic compaction in body order
    if body_flag != 0:
      scan_offset = int(0)
      for w in range(warp):
        scan_offset += misc_sh[w]
      body_awake_ind_out[worldid, scan_offset + _popc(body_mask & lanemask)] = tid
    if tid == 0:
      nbody_awake_out[worldid] = misc_sh[0] + misc_sh[1] + misc_sh[2] + misc_sh[3]

    # ---------------------------------------------------------------- P1d: update_sleep (dofs)
    dof_flag = int(0)
    if is_dof:
      if d_tree >= 0 and body_awake_sh[d_body] == SleepState.AWAKE:
        dof_flag = 1
    dof_mask = _ballot(dof_flag)
    if lane == 0:
      _st_misc_i(misc_sh, 4 + warp, _popc(dof_mask))
    _sync()
    if dof_flag != 0:
      scan_offset = int(0)
      for w in range(warp):
        scan_offset += misc_sh[4 + w]
      dof_awake_ind_out[worldid, scan_offset + _popc(dof_mask & lanemask)] = tid
    if tid == 0:
      nv_awake_out[worldid] = misc_sh[4] + misc_sh[5] + misc_sh[6] + misc_sh[7]

    # ----------------------------------------------------------- P2: kinematics (level-synchronous)
    xanchor = wp.vec3(0.0)
    xaxis = wp.vec3(0.0)
    xpos = wp.vec3(0.0)
    xquat = wp.quat(1.0, 0.0, 0.0, 0.0)
    for level in range(1, nlevel):
      if is_body and b_level == level:
        if b_jntnum == 1 and j_type == JointType.FREE:
          xpos = f_pos
          xquat = wp.normalize(f_quat)
          xanchor = xpos
          xaxis = j_axis
        else:
          if b_mocap >= 0:
            xpos = mocap_pos_in[worldid, b_mocap]
            xquat = mocap_quat_in[worldid, b_mocap]
          else:
            xpos = b_pos
            xquat = b_quat
          pquat = xquat_sh[b_parent]
          xpos = math.rot_vec_quat(xpos, pquat) + xpos_sh[b_parent]
          xquat = math.mul_quat(pquat, xquat)
          if b_jntnum == 1:
            xanchor = math.rot_vec_quat(j_pos, xquat) + xpos
            xaxis = math.rot_vec_quat(j_axis, xquat)
            if j_type == JointType.SLIDE:
              xpos += xaxis * (j_q - j_q0)
            elif j_type == JointType.HINGE:
              xquat = math.mul_quat(xquat, math.axis_angle_to_quat(j_axis, j_q - j_q0))
              # correct for off-center rotation
              xpos = xanchor - math.rot_vec_quat(j_pos, xquat)
          xquat = wp.normalize(xquat)
        _st_body_v3(xpos_sh, tid, xpos)
        _st_body_q(xquat_sh, tid, xquat)
      _sync()

    # ------------------------------------------------------- P2b: body frames, joints, geoms, sites
    xipos = wp.vec3(0.0)
    if is_body:
      if tid == 0:
        xpos = xpos_sh[0]
        xquat = xquat_sh[0]
      xipos = xpos + math.rot_vec_quat(body_ipos[worldid % body_ipos.shape[0], tid], xquat)
      ximat = math.quat_to_mat(math.mul_quat(xquat, body_iquat[worldid % body_iquat.shape[0], tid]))
      xpos_out[worldid, tid] = xpos
      xquat_out[worldid, tid] = xquat
      xmat_out[worldid, tid] = math.quat_to_mat(xquat)
      xipos_out[worldid, tid] = xipos
      ximat_out[worldid, tid] = ximat
      _st_body_v3(xipos_sh, tid, xipos)
      _st_body_v3(com_sh, tid, xipos * b_mass)
      if b_jntnum == 1:
        xanchor_out[worldid, b_jntadr] = xanchor
        xaxis_out[worldid, b_jntadr] = xaxis
    for geomid in range(tid, ngeom, BLOCK):
      bodyid = geom_bodyid[geomid]
      # geoms attached to the world are static (unless descended from mocap bodies)
      if body_weldid[bodyid] != 0 or body_mocapid[body_rootid[bodyid]] != -1:
        gpos = xpos_sh[bodyid]
        gquat = xquat_sh[bodyid]
        geom_xpos_out[worldid, geomid] = gpos + math.rot_vec_quat(geom_pos[worldid % geom_pos.shape[0], geomid], gquat)
        geom_xmat_out[worldid, geomid] = math.quat_to_mat(math.mul_quat(gquat, geom_quat[worldid % geom_quat.shape[0], geomid]))
    for siteid in range(tid, nsite, BLOCK):
      bodyid = site_bodyid[siteid]
      gpos = xpos_sh[bodyid]
      gquat = xquat_sh[bodyid]
      site_xpos_out[worldid, siteid] = gpos + math.rot_vec_quat(site_pos[worldid % site_pos.shape[0], siteid], gquat)
      site_xmat_out[worldid, siteid] = math.quat_to_mat(math.mul_quat(gquat, site_quat[worldid % site_quat.shape[0], siteid]))
    _sync()

    # ---------------------------------------------------------------- P3: subtree_com
    com = wp.vec3(0.0)
    if is_body:
      com = com_sh[tid]
      for c in range(tid + 1, subtree_end):
        com += com_sh[c]
      mass = body_subtreemass[worldid % body_subtreemass.shape[0], tid]
      if mass != 0.0:
        com = com / mass
    _sync()
    if is_body:
      _st_body_v3(com_sh, tid, com)
      subtree_com_out[worldid, tid] = com
      # gravity compensation force of this body (zero unless compensated)
      gforce = wp.vec3(0.0)
      if gravity_enabled and b_gravcomp != 0.0:
        gforce = -gravity * b_mass * b_gravcomp
      _st_body_v3(gforce_sh, tid, gforce)
    _sync()

    # ---------------------------------------------------------------- P3b: cinert, cdof
    if is_body:
      mat = math.quat_to_mat(math.mul_quat(xquat, body_iquat[worldid % body_iquat.shape[0], tid]))
      inert = body_inertia[worldid % body_inertia.shape[0], tid]
      dif = xipos - com_sh[b_root]
      # express inertia in com-based frame (mju_inertCom)
      res = vec10()
      tmp = mat @ wp.diag(inert) @ wp.transpose(mat)
      res[0] = tmp[0, 0]
      res[1] = tmp[1, 1]
      res[2] = tmp[2, 2]
      res[3] = tmp[0, 1]
      res[4] = tmp[0, 2]
      res[5] = tmp[1, 2]
      res[0] += b_mass * (dif[1] * dif[1] + dif[2] * dif[2])
      res[1] += b_mass * (dif[0] * dif[0] + dif[2] * dif[2])
      res[2] += b_mass * (dif[0] * dif[0] + dif[1] * dif[1])
      res[3] -= b_mass * dif[0] * dif[1]
      res[4] -= b_mass * dif[0] * dif[2]
      res[5] -= b_mass * dif[1] * dif[2]
      res[6] = b_mass * dif[0]
      res[7] = b_mass * dif[1]
      res[8] = b_mass * dif[2]
      res[9] = b_mass
      _st_body_v10(cinert_sh, tid, res)
      cinert_out[worldid, tid] = res
      if b_jntnum == 1:
        xmat = wp.transpose(math.quat_to_mat(xquat))
        offset = com_sh[b_root] - xanchor
        if j_type == JointType.FREE:
          _st_dof_sv(cdof_sh, b_dofadr + 0, wp.spatial_vector(0.0, 0.0, 0.0, 1.0, 0.0, 0.0))
          _st_dof_sv(cdof_sh, b_dofadr + 1, wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 1.0, 0.0))
          _st_dof_sv(cdof_sh, b_dofadr + 2, wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 1.0))
          _st_dof_sv(cdof_sh, b_dofadr + 3, wp.spatial_vector(xmat[0], wp.cross(xmat[0], offset)))
          _st_dof_sv(cdof_sh, b_dofadr + 4, wp.spatial_vector(xmat[1], wp.cross(xmat[1], offset)))
          _st_dof_sv(cdof_sh, b_dofadr + 5, wp.spatial_vector(xmat[2], wp.cross(xmat[2], offset)))
        elif j_type == JointType.SLIDE:
          _st_dof_sv(cdof_sh, b_dofadr, wp.spatial_vector(wp.vec3(0.0), xaxis))
        else:  # HINGE
          _st_dof_sv(cdof_sh, b_dofadr, wp.spatial_vector(xaxis, wp.cross(xaxis, offset)))
    _sync()

    # ---------------------------------------------------------------- P4: crb and M
    crb = vec10()
    if is_body:
      crb = cinert_sh[tid]
      # the world body never accumulates its children
      if tid > 0:
        for c in range(tid + 1, subtree_end):
          crb += cinert_sh[c]
      crb_out[worldid, tid] = crb
      _st_body_v10(crb_sh, tid, crb)
    cdof = wp.spatial_vector()
    if is_dof:
      cdof = cdof_sh[tid]
      cdof_out[worldid, tid] = cdof
    _sync()
    if is_dof:
      rownnz = M_rownnz[tid]
      madr_ij = M_rowadr[tid] + rownnz - 1
      buf = math.inert_vec(crb_sh[d_body], cdof)
      # diagonal: armature inertia plus the composite inertia term
      M_out[worldid, madr_ij] = dof_armature[worldid % dof_armature.shape[0], tid] + wp.dot(cdof, buf)
      # ancestors fill the rest of the CSR row; "simple" dofs (rownnz == 1) keep their in-joint
      # parent chain but store only the diagonal, so the walk is bounded by the row length
      dofid = int(tid)
      for _k in range(rownnz - 1):
        madr_ij -= 1
        dofid = dof_parent_sh[dofid]
        M_out[worldid, madr_ij] = wp.dot(cdof_sh[dofid], buf)

    # ---------------------------------------------------------- P5: transmission, actuator velocity
    for actid in range(tid, nu, BLOCK):
      jntid = actuator_trnid[actid][0]
      gear0 = actuator_gear[worldid % actuator_gear.shape[0], actid][0]
      qadr = jnt_qposadr[jntid]
      vadr = jnt_dofadr[jntid]
      actuator_length_out[worldid, actid] = qpos_in[worldid, qadr] * gear0
      moment_rownnz_out[worldid, actid] = 1
      moment_rowadr_out[worldid, actid] = actid
      moment_colind_out[worldid, actid] = vadr
      actuator_moment_out[worldid, actid] = gear0
      actuator_velocity_out[worldid, actid] = gear0 * qvel_sh[vadr]
      _st_act_i(act_dof_sh, actid, vadr)

    # ----------------------------------------------------- P6: com_vel and cacc (level-synchronous)
    if tid == 0:
      _st_body_sv(cvel_sh, 0, wp.spatial_vector())
      cacc0 = wp.spatial_vector()
      if gravity_enabled:
        cacc0 = wp.spatial_vector(wp.vec3(0.0), -gravity)
      _st_body_sv(cacc_sh, 0, cacc0)
    _sync()
    for level in range(1, nlevel):
      if is_body and b_level == level:
        cvel = cvel_sh[b_parent]
        cacc = cacc_sh[b_parent]
        if b_jntnum == 1:
          if j_type == JointType.FREE:
            for k in range(3):
              cvel += cdof_sh[b_dofadr + k] * qvel_sh[b_dofadr + k]
              cdof_dot_out[worldid, b_dofadr + k] = wp.spatial_vector()
            for k in range(3, 6):
              cdof_dot = math.motion_cross(cvel, cdof_sh[b_dofadr + k])
              cdof_dot_out[worldid, b_dofadr + k] = cdof_dot
              cacc += cdof_dot * qvel_sh[b_dofadr + k]
            for k in range(3, 6):
              cvel += cdof_sh[b_dofadr + k] * qvel_sh[b_dofadr + k]
          else:
            cdof_dot = math.motion_cross(cvel, cdof_sh[b_dofadr])
            cdof_dot_out[worldid, b_dofadr] = cdof_dot
            cacc += cdof_dot * qvel_sh[b_dofadr]
            cvel += cdof_sh[b_dofadr] * qvel_sh[b_dofadr]
        _st_body_sv(cvel_sh, tid, cvel)
        _st_body_sv(cacc_sh, tid, cacc)
      _sync()
    if is_body:
      cvel_out[worldid, tid] = cvel_sh[tid]
      cacc_out[worldid, tid] = cacc_sh[tid]

    # ---------------------------------------------------------------- P7: passive (per dof)
    qfrc_spring = float(0.0)
    qfrc_damper = float(0.0)
    qfrc_gravcomp = float(0.0)
    qfrc_passive = float(0.0)
    if is_dof:
      if not (dsbl_spring and dsbl_damper):
        jnttype = jnt_type[d_jnt]
        jdof = jnt_dofadr[d_jnt]
        qposid = jnt_qposadr[d_jnt]
        stiffness = jnt_stiffness[worldid % jnt_stiffness.shape[0], d_jnt]
        spoly = jnt_stiffnesspoly[worldid % jnt_stiffnesspoly.shape[0], d_jnt]
        # the joint's first dof supplies the damping of all its dofs (as _spring_damper_dof_passive)
        damping = dof_damping[worldid % dof_damping.shape[0], jdof]
        dpoly = dof_dampingpoly[worldid % dof_dampingpoly.shape[0], jdof]
        has_stiffness = (stiffness != 0.0 or spoly[0] != 0.0 or spoly[1] != 0.0) and not dsbl_spring
        has_damping = (damping != 0.0 or dpoly[0] != 0.0 or dpoly[1] != 0.0) and not dsbl_damper
        qpos_spring_id = worldid % qpos_spring.shape[0]
        if jnttype == JointType.FREE:
          if has_stiffness:
            k = tid - jdof
            if k < 3:
              dif = wp.vec3(
                qpos_in[worldid, qposid + 0] - qpos_spring[qpos_spring_id, qposid + 0],
                qpos_in[worldid, qposid + 1] - qpos_spring[qpos_spring_id, qposid + 1],
                qpos_in[worldid, qposid + 2] - qpos_spring[qpos_spring_id, qposid + 2],
              )
              kf = util_misc._poly_force(stiffness, spoly, wp.length(dif), 0)
              qfrc_spring = -kf * dif[k]
            else:
              rot = wp.quat(
                qpos_in[worldid, qposid + 3],
                qpos_in[worldid, qposid + 4],
                qpos_in[worldid, qposid + 5],
                qpos_in[worldid, qposid + 6],
              )
              rot = wp.normalize(rot)
              ref = wp.quat(
                qpos_spring[qpos_spring_id, qposid + 3],
                qpos_spring[qpos_spring_id, qposid + 4],
                qpos_spring[qpos_spring_id, qposid + 5],
                qpos_spring[qpos_spring_id, qposid + 6],
              )
              dif = math.quat_sub(rot, ref)
              k_rot = util_misc._poly_force(stiffness, spoly, wp.length(dif), 0)
              qfrc_spring = -k_rot * dif[k - 3]
          if has_damping:
            qfrc_damper = -d_qvel * util_misc._poly_force(damping, dpoly, d_qvel, 1)
        else:  # SLIDE, HINGE
          if has_stiffness:
            fdif = qpos_in[worldid, qposid] - qpos_spring[qpos_spring_id, qposid]
            qfrc_spring = -fdif * util_misc._poly_force(stiffness, spoly, fdif, 0)
          if has_damping:
            qfrc_damper = -d_qvel * util_misc._poly_force(damping, dpoly, d_qvel, 1)

        # gravity compensation: bodies in this dof's subtree (support.jac_dof via
        # body_isdofancestor)
        if gravity_enabled:
          cdof_ang = wp.spatial_top(cdof)
          cdof_lin = wp.spatial_bottom(cdof)
          dof_send = send_sh[d_body]
          for g in range(ngravcomp):
            bodyid = gc_body_sh[g]
            if bodyid >= d_body and bodyid < dof_send:
              offset = xipos_sh[bodyid] - com_sh[body_rootid[bodyid]]
              jacp = cdof_lin + wp.cross(cdof_ang, offset)
              qfrc_gravcomp += wp.dot(jacp, gforce_sh[bodyid])

        qfrc_passive = qfrc_spring + qfrc_damper
        if gravity_enabled:
          if jnt_actgravcomp[d_jnt] == 0:
            qfrc_passive += qfrc_gravcomp

      qfrc_spring_out[worldid, tid] = qfrc_spring
      qfrc_damper_out[worldid, tid] = qfrc_damper
      qfrc_gravcomp_out[worldid, tid] = qfrc_gravcomp
      qfrc_adhesion_out[worldid, tid] = 0.0
      qfrc_passive_out[worldid, tid] = qfrc_passive

    # ---------------------------------------------------------------- P8: rne (cfrc_int, qfrc_bias)
    if is_body:
      frc = wp.spatial_vector()
      if tid > 0:
        cinert = cinert_sh[tid]
        cvel = cvel_sh[tid]
        frc = math.inert_vec(cinert, cacc_sh[tid])
        frc += math.motion_cross_force(cvel, math.inert_vec(cinert, cvel))
      # the cacc slot now holds the body-local force
      _st_body_sv(cacc_sh, tid, frc)
    _sync()
    cfrc = wp.spatial_vector()
    if is_body:
      cfrc = cacc_sh[tid]
      for c in range(tid + 1, subtree_end):
        cfrc += cacc_sh[c]
    _sync()
    if is_body:
      _st_body_sv(cacc_sh, tid, cfrc)
      cfrc_int_out[worldid, tid] = cfrc
    _sync()
    if is_dof:
      qfrc_bias_out[worldid, tid] = wp.dot(cdof, cacc_sh[d_body])

    # ---------------------------------------------------------------- P9: actuation
    if nu == 0 or (opt_disableflags & DisableBit.ACTUATION) != 0:
      for actid in range(tid, nu, BLOCK):
        actuator_force_out[worldid, actid] = 0.0
      if is_dof:
        qfrc_actuator_out[worldid, tid] = 0.0
    else:
      for actid in range(tid, nu, BLOCK):
        ctrl = ctrl_in[worldid, actid]
        if actuator_ctrllimited[actid] and (opt_disableflags & DisableBit.CLAMPCTRL) == 0:
          ctrlrange = actuator_ctrlrange[worldid % actuator_ctrlrange.shape[0], actid]
          ctrl = wp.clamp(ctrl, ctrlrange[0], ctrlrange[1])
        jntid = actuator_trnid[actid][0]
        gear0 = actuator_gear[worldid % actuator_gear.shape[0], actid][0]
        length = qpos_in[worldid, jnt_qposadr[jntid]] * gear0
        velocity = gear0 * qvel_sh[act_dof_sh[actid]]

        gainprm = actuator_gainprm[worldid % actuator_gainprm.shape[0], actid]
        gain = gainprm[0]
        if actuator_gaintype[actid] == GainType.AFFINE:
          gain = gainprm[0] + gainprm[1] * length + gainprm[2] * velocity

        bias = float(0.0)
        if actuator_biastype[actid] == BiasType.AFFINE:
          biasprm = actuator_biasprm[worldid % actuator_biasprm.shape[0], actid]
          bias = biasprm[0] + biasprm[1] * length + biasprm[2] * velocity

        force = gain * ctrl + bias
        if actuator_forcelimited[actid]:
          forcerange = actuator_forcerange[worldid % actuator_forcerange.shape[0], actid]
          force = wp.clamp(force, forcerange[0], forcerange[1])
        actuator_force_out[worldid, actid] = force
        # moment of a joint transmission is the gear ratio
        _st_act_f(act_force_sh, actid, force * gear0)
      _sync()
      if is_dof:
        qfrc = float(0.0)
        for actid in range(nu):
          if act_dof_sh[actid] == tid:
            qfrc += act_force_sh[actid]
        # actuator-level gravity compensation, skip if added as passive force
        if gravity_enabled and jnt_actgravcomp[d_jnt] != 0:
          qfrc += qfrc_gravcomp
        if jnt_actfrclimited[d_jnt]:
          frcrange = jnt_actfrcrange[worldid % jnt_actfrcrange.shape[0], d_jnt]
          qfrc = wp.clamp(qfrc, frcrange[0], frcrange[1])
        qfrc_actuator_out[worldid, tid] = qfrc

  return kernel


@cache_kernel
def _forward_b_kernel(NB: int, NV: int, NT: int, NSMALL: int, NBIGDOF: int, NBIG: int):
  BLOCK = NV
  # dense scratch slot per tree: n*n factor followed by n solution entries
  SMALL_SLOT = NSMALL * NSMALL + NSMALL
  BIG_SLOT = NBIGDOF * NBIGDOF + NBIGDOF
  FAC_SIZE = NT * SMALL_SLOT + NBIG * BIG_SLOT

  @wp.func_native(snippet="values.data(wp::tile_coord(index)) = value;")
  def _st_fac(values: wp.tile[float, FAC_SIZE], index: int, value: float): ...

  @wp.func_native(snippet="values.data(wp::tile_coord(index)) = value;")
  def _st_body_sv(values: wp.tile[wp.spatial_vector, NB], index: int, value: wp.spatial_vector): ...

  @wp.func_native(snippet="values.data(wp::tile_coord(index)) = value;")
  def _st_tree_i(values: wp.tile[int, NT], index: int, value: int): ...

  @wp.func_native(snippet="values.data(wp::tile_coord(index)) = value;")
  def _st_misc_i(values: wp.tile[int, 4], index: int, value: int): ...

  @wp.kernel(module="unique", enable_backward=False, grid_stride=False, launch_bounds=(BLOCK,))
  def kernel(
    # Model:
    nbody: int,
    nv: int,
    ntree: int,
    body_parentid: wp.array[int],
    body_rootid: wp.array[int],
    body_treeid: wp.array[int],
    dof_bodyid: wp.array[int],
    dof_treeid: wp.array[int],
    tree_dofadr: wp.array[int],
    tree_dofnum: wp.array[int],
    M_rownnz: wp.array[int],
    M_rowadr: wp.array[int],
    M_colind: wp.array[int],
    qLD_block_adr: wp.array[int],
    # Data in:
    nvmax_in: int,
    nvmax_pad_in: int,
    tile_size_in: int,
    warn_overflow: bool,
    tree_awake_in: wp.array2d[int],
    tree_island_in: wp.array2d[int],
    island_nv_in: wp.array2d[int],
    qfrc_passive_in: wp.array2d[float],
    qfrc_bias_in: wp.array2d[float],
    qfrc_actuator_in: wp.array2d[float],
    qfrc_applied_in: wp.array2d[float],
    xfrc_applied_in: wp.array2d[wp.spatial_vector],
    xipos_in: wp.array2d[wp.vec3],
    subtree_com_in: wp.array2d[wp.vec3],
    cdof_in: wp.array2d[wp.spatial_vector],
    M_in: wp.array2d[float],
    # Data out:
    qfrc_smooth_out: wp.array2d[float],
    qLD_out: wp.array2d[float],
    qLDiagInv_out: wp.array2d[float],
    qacc_smooth_out: wp.array2d[float],
    dof_cdof_out: wp.array2d[int],
    cdof_dof_out: wp.array2d[int],
    ncdof_out: wp.array[int],
    nsingleton6_out: wp.array[int],
    overflow_out: wp.array[int],
  ):
    worldid, tid = wp.tid()

    fac_sh = wp.tile_empty(shape=(FAC_SIZE,), dtype=float, storage="shared")
    xfrc_sh = wp.tile_empty(shape=(NB,), dtype=wp.spatial_vector, storage="shared")
    tree_awake_sh = wp.tile_empty(shape=(NT,), dtype=int, storage="shared")
    tree_island_sh = wp.tile_empty(shape=(NT,), dtype=int, storage="shared")
    island_nv_sh = wp.tile_empty(shape=(NT,), dtype=int, storage="shared")
    tree_dofnum_sh = wp.tile_empty(shape=(NT,), dtype=int, storage="shared")
    tree_base_sh = wp.tile_empty(shape=(NT,), dtype=int, storage="shared")
    layout_sh = wp.tile_empty(shape=(4,), dtype=int, storage="shared")

    is_body = tid < nbody
    is_dof = tid < nv
    is_tree = tid < ntree

    # ---------------------------------------------------------- Q0: loads, map reset, scratch slots
    if is_body:
      _st_body_sv(xfrc_sh, tid, xfrc_applied_in[worldid, tid])
    t_n = int(0)
    t_start = int(0)
    t_compact = int(0)
    if is_tree:
      t_n = tree_dofnum[tid]
      t_start = tree_dofadr[tid]
      _st_tree_i(tree_awake_sh, tid, tree_awake_in[worldid, tid])
      _st_tree_i(tree_island_sh, tid, tree_island_in[worldid, tid])
      _st_tree_i(island_nv_sh, tid, island_nv_in[worldid, tid])
      _st_tree_i(tree_dofnum_sh, tid, t_n)
      if qLD_block_adr[t_start] == Q_LD_BLOCK_COMPACT:
        t_compact = 1
      # dense scratch slot: small trees share a fixed-size slot table, wide trees follow it
      base = tid * wp.static(SMALL_SLOT)
      if t_n > wp.static(NSMALL):
        nbig = int(0)
        for t in range(tid):
          if tree_dofnum[t] > wp.static(NSMALL):
            nbig += 1
        base = wp.static(NT * SMALL_SLOT) + nbig * wp.static(BIG_SLOT)
      _st_tree_i(tree_base_sh, tid, base)
      if t_compact == 0:
        for i in range(t_n * t_n):
          _st_fac(fac_sh, base + i, 0.0)
    d_body = int(0)
    d_tree = int(-1)
    d_rowadr = int(0)
    d_rownnz = int(0)
    d_block_adr = int(0)
    if is_dof:
      d_body = dof_bodyid[tid]
      d_tree = dof_treeid[tid]
      d_rowadr = M_rowadr[tid]
      d_rownnz = M_rownnz[tid]
      d_block_adr = qLD_block_adr[tid]
      dof_cdof_out[worldid, tid] = -1
    for i in range(tid, nvmax_pad_in, BLOCK):
      cdof_dof_out[worldid, i] = -1
    _sync()

    # ------------------------------------------------------------ Q1: qfrc_smooth + xfrc_accumulate
    qfrc = float(0.0)
    if is_dof:
      if d_tree < 0 or tree_awake_sh[d_tree] != 0:
        qfrc = (
          qfrc_passive_in[worldid, tid]
          - qfrc_bias_in[worldid, tid]
          + qfrc_actuator_in[worldid, tid]
          + qfrc_applied_in[worldid, tid]
        )
      # J^T xfrc_applied over the bodies in this dof's subtree (support._apply_ft)
      cdof = cdof_in[worldid, tid]
      rotational_cdof = wp.vec3(cdof[0], cdof[1], cdof[2])
      jac = wp.spatial_vector(cdof[3], cdof[4], cdof[5], cdof[0], cdof[1], cdof[2])
      accumul = float(0.0)
      for bodyid in range(d_body, nbody):
        ft_body = xfrc_sh[bodyid]
        if ft_body == wp.spatial_vector():
          continue
        parentid = bodyid
        while parentid != 0 and parentid != d_body:
          parentid = body_parentid[parentid]
        if parentid == 0:
          continue
        offset = xipos_in[worldid, bodyid] - subtree_com_in[worldid, body_rootid[bodyid]]
        cross_term = wp.cross(rotational_cdof, offset)
        accumul += wp.dot(jac, ft_body) + wp.dot(cross_term, wp.spatial_top(ft_body))
      qfrc += accumul
      qfrc_smooth_out[worldid, tid] = qfrc

      # gather this dof's CSR row into the tree's dense scratch (columns are ancestors); compact
      # (diagonal) blocks solve directly with the reciprocal diagonal
      if d_block_adr == Q_LD_BLOCK_COMPACT:
        inverse = 1.0 / M_in[worldid, d_rowadr]
        qLDiagInv_out[worldid, tid] = inverse
        x = inverse * qfrc
        if tree_awake_sh[d_tree] == 0:
          x = 0.0
        qacc_smooth_out[worldid, tid] = x
      else:
        base = tree_base_sh[d_tree]
        start = tree_dofadr[d_tree]
        n = tree_dofnum_sh[d_tree]
        i = tid - start
        for k in range(d_rownnz):
          col = M_colind[d_rowadr + k] - start
          _st_fac(fac_sh, base + i * n + col, M_in[worldid, d_rowadr + k])
        _st_fac(fac_sh, base + n * n + i, qfrc)
    _sync()

    # ---------------------------------------------------------- Q2: per-tree dense factor and solve
    if is_tree and t_compact == 0:
      n = t_n
      base = tree_base_sh[tid]
      xbase = base + n * n
      awake = tree_awake_sh[tid]
      factor_adr = qLD_block_adr[t_start]

      # upper Cholesky M = U^T U with the solve interleaved
      # (smooth._small_cholesky_factorize_solve_block)
      for i in range(n):
        diagonal_value = fac_sh[base + i * n + i]
        rhs_value = fac_sh[xbase + i]
        for k in range(i):
          factor = fac_sh[base + k * n + i]
          diagonal_value -= factor * factor
          rhs_value -= factor * fac_sh[xbase + k]
        diagonal_factor = wp.sqrt(diagonal_value)
        _st_fac(fac_sh, base + i * n + i, diagonal_factor)
        diagonal_inv = 1.0 / diagonal_factor
        _st_fac(fac_sh, xbase + i, rhs_value * diagonal_inv)
        for j in range(i + 1, n):
          value = fac_sh[base + j * n + i]
          for k in range(i):
            value -= fac_sh[base + k * n + i] * fac_sh[base + k * n + j]
          _st_fac(fac_sh, base + i * n + j, value * diagonal_inv)

      for reverse_i in range(n):
        i = n - 1 - reverse_i
        value = fac_sh[xbase + i]
        for k in range(i + 1, n):
          value -= fac_sh[base + i * n + k] * fac_sh[xbase + k]
        _st_fac(fac_sh, xbase + i, value / fac_sh[base + i * n + i])

      # publish the packed upper factor (the unused lower triangle reads as zero) and the solution
      for i in range(n):
        for j in range(n):
          value = float(0.0)
          if j >= i:
            value = fac_sh[base + i * n + j]
          qLD_out[worldid, factor_adr + i * n + j] = value
        x = fac_sh[xbase + i]
        if awake == 0:
          x = 0.0
        qacc_smooth_out[worldid, t_start + i] = x

    # ------------------------------------------- Q3: compaction layout (island._compact_dof_layout)
    if tid == 0:
      count = int(0)
      singleton_count = int(0)
      for t in range(ntree):
        island_id = tree_island_sh[t]
        if tree_awake_sh[t] == 1 and island_id >= 0:
          num = tree_dofnum_sh[t]
          count += num
          if num == 6 and island_nv_sh[island_id] == 6:
            singleton_count += 1

      if count > nvmax_in:
        if warn_overflow:
          wp.printf(
            "nvmax overflow: world %d needs %d active DOFs but nvmax = %d (behavior undefined)\n",
            worldid,
            count,
            nvmax_in,
          )
        overflow_out[worldid] = overflow_out[worldid] | OverflowType.NVMAX
        ncdof_out[worldid] = nvmax_in
        nsingleton6_out[worldid] = 0
        _st_misc_i(layout_sh, 0, nvmax_in)
        _st_misc_i(layout_sh, 1, 0)
      else:
        general_count = count - 6 * singleton_count
        general_end = int(0)
        if general_count > 0:
          general_end = ((general_count + tile_size_in - 1) // tile_size_in) * tile_size_in
        # keep the final slot free for the augmented Cholesky column; fold six-DOF trees back into
        # the aligned general block only as far as necessary
        while singleton_count > 0 and general_end + 6 * singleton_count >= nvmax_pad_in:
          singleton_count -= 1
          general_count += 6
          general_end = ((general_count + tile_size_in - 1) // tile_size_in) * tile_size_in
        ncdof_out[worldid] = count
        nsingleton6_out[worldid] = singleton_count
        _st_misc_i(layout_sh, 0, count)
        _st_misc_i(layout_sh, 1, singleton_count)
    _sync()

    # ---------------------------------------------- Q3b: compaction maps (island._map_compact_dofs)
    if is_tree:
      if tree_awake_sh[tid] != 0 and tree_island_sh[tid] >= 0:
        singleton_count = layout_sh[1]
        general_count = layout_sh[0] - 6 * singleton_count
        general_end = int(0)
        if general_count > 0:
          general_end = ((general_count + tile_size_in - 1) // tile_size_in) * tile_size_in

        # prefix counts preserve tree order while each tree maps independently
        general = int(0)
        singleton = int(0)
        for t in range(tid):
          island_id = tree_island_sh[t]
          if tree_awake_sh[t] == 0 or island_id < 0:
            continue
          is_singleton = singleton < singleton_count and tree_dofnum_sh[t] == 6 and island_nv_sh[island_id] == 6
          if is_singleton:
            singleton += 1
          else:
            general += tree_dofnum_sh[t]

        island_id = tree_island_sh[tid]
        is_singleton = singleton < singleton_count and t_n == 6 and island_nv_sh[island_id] == 6
        start = general
        if is_singleton:
          start = general_end + 6 * singleton
        for j in range(t_n):
          compact = start + j
          if singleton_count > 0 or compact < nvmax_in:
            dof = t_start + j
            dof_cdof_out[worldid, dof] = compact
            cdof_dof_out[worldid, compact] = dof

  return kernel


# ------------------------------------------------------------------------------------------------
# Per-world contact grouping: Newton assigns contact ids with a global atomic counter, so the
# contacts of one world are scattered over [0, nacon). A counting sort by world (count, then prefix
# + scatter) gives every CTA a contiguous id list. Contacts may be appended every substep (wake
# injection), so the grouping runs per substep. The slot order within a world follows the scatter
# atomics, i.e. the contact row order is unspecified exactly as with the stock atomic row
# allocation.
# ------------------------------------------------------------------------------------------------

# fixed grid for the grouping launches (they grid-stride over nacon, which is a device scalar)
_GROUP_THREADS = 65536


@wp.kernel(enable_backward=False, grid_stride=False)
def _world_contact_count(
  # Data in:
  nacon_in: wp.array[int],
  contact_worldid_in: wp.array[int],
  # In:
  total_threads: int,
  # Out:
  count_out: wp.array[int],
):
  tid = wp.tid()
  n = wp.min(nacon_in[0], contact_worldid_in.shape[0])
  for cid in range(tid, n, total_threads):
    wp.atomic_add(count_out, contact_worldid_in[cid], 1)


@wp.kernel(enable_backward=False, grid_stride=False, launch_bounds=(NV_CAP,))
def _world_contact_scatter(
  # Data in:
  nworld: int,
  nacon_in: wp.array[int],
  contact_worldid_in: wp.array[int],
  # In:
  count_in: wp.array[int],
  total_threads: int,
  # Out:
  fill_out: wp.array[int],
  start_out: wp.array[int],
  list_out: wp.array[int],
):
  tid = wp.tid()
  local = _thread_index()
  lane = local & 31
  warp = local >> 5
  chunk_sh = wp.tile_empty(shape=(NV_CAP,), dtype=int, storage="shared")
  scan_sh = wp.tile_empty(shape=(8,), dtype=int, storage="shared")

  # every block recomputes the exclusive world prefix: thread-local chunk sums, block scan
  chunk = (nworld + NV_CAP - 1) // NV_CAP
  begin = wp.min(local * chunk, nworld)
  end = wp.min(begin + chunk, nworld)
  total = int(0)
  for w in range(begin, end):
    total += count_in[w]
  prefix = _block_scan(scan_sh, total, lane, warp)
  _st_chunk_i(chunk_sh, local, prefix)
  if tid < NV_CAP:
    # the first block publishes the per-world starts for the consumer kernel
    running = int(prefix)
    for w in range(begin, end):
      start_out[w] = running
      running += count_in[w]
  _sync()

  n = wp.min(nacon_in[0], contact_worldid_in.shape[0])
  for cid in range(tid, n, total_threads):
    worldid = contact_worldid_in[cid]
    c = worldid // chunk
    base = chunk_sh[c]
    for w in range(c * chunk, worldid):
      base += count_in[w]
    slot = wp.atomic_add(fill_out, worldid, 1)
    list_out[base + slot] = cid


# resident CTAs per SM requested from the compiler for the register-bound middle kernel (two waves
# at 1024 worlds on 170 SMs need >= 4 CTAs/SM)
_FORWARD_M_MIN_BLOCKS = 4


@cache_kernel
def _forward_m_kernel(NB: int, NV: int, NT: int):
  BLOCK = NV

  @wp.kernel(module="unique", enable_backward=False, grid_stride=False, launch_bounds=(BLOCK, _FORWARD_M_MIN_BLOCKS))
  def kernel(
    # Model:
    nv: int,
    nbody: int,
    ntree: int,
    neq: int,
    neq_jnt: int,
    nlimit: int,
    opt_timestep: wp.array[float],
    opt_disableflags: int,
    opt_impratio_invsqrt: wp.array[float],
    qpos0: wp.array2d[float],
    body_rootid: wp.array[int],
    body_weldid: wp.array[int],
    body_mocapid: wp.array[int],
    body_treeid: wp.array[int],
    body_dofnum: wp.array[int],
    body_dofadr: wp.array[int],
    body_invweight0: wp.array2d[wp.vec2],
    jnt_qposadr: wp.array[int],
    jnt_dofadr: wp.array[int],
    jnt_bodyid: wp.array[int],
    jnt_solref: wp.array2d[wp.vec2],
    jnt_solimp: wp.array2d[vec5],
    jnt_range: wp.array2d[wp.vec2],
    jnt_margin: wp.array2d[float],
    jnt_limited_slide_hinge_adr: wp.array[int],
    dof_bodyid: wp.array[int],
    dof_parentid: wp.array[int],
    dof_treeid: wp.array[int],
    dof_solref: wp.array2d[wp.vec2],
    dof_solimp: wp.array2d[vec5],
    dof_frictionloss: wp.array2d[float],
    dof_invweight0: wp.array2d[float],
    tree_dofnum: wp.array[int],
    geom_bodyid: wp.array[int],
    eq_obj1id: wp.array[int],
    eq_obj2id: wp.array[int],
    eq_solref: wp.array2d[wp.vec2],
    eq_solimp: wp.array2d[vec5],
    eq_data: wp.array2d[vec11],
    eq_jnt_adr: wp.array[int],
    # Data in:
    njmax_in: int,
    njmax_nnz_in: int,
    qpos_in: wp.array2d[float],
    qvel_in: wp.array2d[float],
    eq_active_in: wp.array2d[bool],
    tree_awake_in: wp.array2d[int],
    subtree_com_in: wp.array2d[wp.vec3],
    cdof_in: wp.array2d[wp.spatial_vector],
    contact_dist_in: wp.array[float],
    contact_dim_in: wp.array[int],
    contact_includemargin_in: wp.array[float],
    contact_geom_in: wp.array[wp.vec2i],
    contact_type_in: wp.array[int],
    contact_pos_in: wp.array[wp.vec3],
    contact_frame_in: wp.array2d[wp.vec3],
    contact_friction_in: wp.array[vec5],
    contact_solref_in: wp.array[wp.vec2],
    contact_solimp_in: wp.array[vec5],
    world_con_start_in: wp.array[int],
    world_con_count_in: wp.array[int],
    world_con_list_in: wp.array[int],
    # Data out:
    tree_asleep_out: wp.array2d[int],
    tree_awake_out: wp.array2d[int],
    ntree_awake_out: wp.array[int],
    body_awake_out: wp.array2d[int],
    body_awake_ind_out: wp.array2d[int],
    nbody_awake_out: wp.array[int],
    dof_awake_ind_out: wp.array2d[int],
    nv_awake_out: wp.array[int],
    ne_out: wp.array[int],
    nf_out: wp.array[int],
    nl_out: wp.array[int],
    nefc_out: wp.array[int],
    efc_type_out: wp.array2d[int],
    efc_id_out: wp.array2d[int],
    efc_jtdaj_adr_out: wp.array2d[int],
    efc_jtdaj_nrow_out: wp.array2d[int],
    efc_jtdaj_nblock_out: wp.array[int],
    efc_J_rownnz_out: wp.array2d[int],
    efc_J_rowadr_out: wp.array2d[int],
    efc_J_colind_out: wp.array3d[int],
    efc_J_out: wp.array3d[float],
    efc_pos_out: wp.array2d[float],
    efc_margin_out: wp.array2d[float],
    efc_D_out: wp.array2d[float],
    efc_vel_out: wp.array2d[float],
    efc_aref_out: wp.array2d[float],
    efc_frictionloss_out: wp.array2d[float],
    efc_Jqvel_out: wp.array2d[float],
    contact_efc_address_out: wp.array2d[int],
    overflow_out: wp.array[int],
    nisland_out: wp.array[int],
    tree_island_out: wp.array2d[int],
    island_nv_out: wp.array2d[int],
  ):
    worldid, tid = wp.tid()

    tree_asleep_sh = wp.tile_empty(shape=(NT,), dtype=int, storage="shared")
    edge_sh = wp.tile_empty(shape=(NT,), dtype=wp.uint32, storage="shared")
    body_awake_sh = wp.tile_empty(shape=(NB,), dtype=int, storage="shared")
    misc_sh = wp.tile_empty(shape=(16,), dtype=int, storage="shared")
    scan_sh = wp.tile_empty(shape=(8,), dtype=int, storage="shared")
    # per-chunk contact stash for the row phase
    c_cid_sh = wp.tile_empty(shape=(NV,), dtype=int, storage="shared")
    c_row_sh = wp.tile_empty(shape=(NV,), dtype=int, storage="shared")
    c_rowadr_sh = wp.tile_empty(shape=(NV,), dtype=int, storage="shared")
    c_rownnz_sh = wp.tile_empty(shape=(NV,), dtype=int, storage="shared")
    c_body1_sh = wp.tile_empty(shape=(NV,), dtype=int, storage="shared")
    c_body2_sh = wp.tile_empty(shape=(NV,), dtype=int, storage="shared")

    lane = tid & 31
    warp = tid >> 5
    timestep = opt_timestep[worldid % opt_timestep.shape[0]]
    impratio_invsqrt = opt_impratio_invsqrt[worldid % opt_impratio_invsqrt.shape[0]]

    constraint_on = (opt_disableflags & DisableBit.CONSTRAINT) == 0
    eq_on = constraint_on and (opt_disableflags & DisableBit.EQUALITY) == 0
    fric_on = constraint_on and (opt_disableflags & DisableBit.FRICTIONLOSS) == 0
    lim_on = constraint_on and (opt_disableflags & DisableBit.LIMIT) == 0
    con_on = constraint_on and (opt_disableflags & DisableBit.CONTACT) == 0

    # ---------------------------------------------------------------- M0: loads
    if tid < NT:
      _st_tree32_u(edge_sh, tid, wp.uint32(0))
    if tid < ntree:
      _st_tree32_i(tree_asleep_sh, tid, tree_asleep_out[worldid, tid])
    _sync()

    # -------------------------------------------------------------- M1: sleep.wake_equality (JOINT)
    # one thread walks the equalities in order (stock: one thread per equality with benign races);
    # the awake test reads the pre-wake tree_awake snapshot exactly like the stock kernel
    if tid == 0:
      for eqid in range(neq):
        if not eq_active_in[worldid, eqid]:
          continue
        id1 = eq_obj1id[eqid]
        id2 = eq_obj2id[eqid]
        tree1 = int(-1)
        tree2 = int(-1)
        if id1 >= 0:
          tree1 = body_treeid[jnt_bodyid[id1]]
        if id2 >= 0:
          tree2 = body_treeid[jnt_bodyid[id2]]
        s1 = int(SleepState.STATIC)
        s2 = int(SleepState.STATIC)
        if tree1 >= 0:
          s1 = tree_awake_in[worldid, tree1]
        if tree2 >= 0:
          s2 = tree_awake_in[worldid, tree2]
        if s1 != SleepState.ASLEEP and s2 != SleepState.ASLEEP:
          continue
        if s1 == SleepState.STATIC or s2 == SleepState.STATIC:
          continue
        if tree1 == tree2:
          continue
        if s1 == SleepState.ASLEEP and s2 == SleepState.ASLEEP:
          cycle1 = _sleep_cycle_sh(tree_asleep_sh, ntree, tree1)
          cycle2 = _sleep_cycle_sh(tree_asleep_sh, ntree, tree2)
          if cycle1 != cycle2:
            _wake_tree_sh(tree_asleep_sh, ntree, tree1, _K_AWAKE_VAL)
            _wake_tree_sh(tree_asleep_sh, ntree, tree2, _K_AWAKE_VAL)
        else:
          sleeping_tree = tree2
          if s1 == SleepState.ASLEEP:
            sleeping_tree = tree1
          _wake_tree_sh(tree_asleep_sh, ntree, sleeping_tree, _K_AWAKE_VAL)
    _sync()

    # ---------------------------------------------------------------- M2: sleep.update_sleep
    _publish_sleep_state(
      tree_asleep_sh,
      body_awake_sh,
      misc_sh,
      nbody,
      nv,
      ntree,
      body_rootid,
      body_mocapid,
      body_treeid,
      dof_bodyid,
      worldid,
      tid,
      tree_asleep_out,
      tree_awake_out,
      ntree_awake_out,
      body_awake_out,
      body_awake_ind_out,
      nbody_awake_out,
      dof_awake_ind_out,
      nv_awake_out,
    )

    # ---------------------------------------------------------------- M3: constraint rows
    # canonical order ne | nf | nl | contacts; within a family, rows follow the item order. Row and
    # J-nonzero addresses are exclusive prefixes (stock: atomic allocation). Rows at or beyond njmax
    # are counted in nefc but not written (their J nonzeros are not allocated either), as in stock.
    row_base = int(0)
    nnz_base = int(0)

    # equality (JOINT): one row, one block, nnz 1 or 2
    if eq_on:
      for i0 in range(0, neq_jnt, BLOCK):
        i = i0 + tid
        rows = int(0)
        nnz = int(0)
        eqid = int(-1)
        jntid_2 = int(-1)
        if i < neq_jnt:
          eqid = eq_jnt_adr[i]
          if eq_active_in[worldid, eqid]:
            rows = 1
            jntid_2 = eq_obj2id[eqid]
            nnz = 1
            if jntid_2 > -1:
              nnz = 2
        efcid = row_base + _block_scan(scan_sh, rows, lane, warp)
        rows_total = scan_sh[4]
        rownnz = nnz
        if efcid >= njmax_in:
          nnz = 0
        rowadr = nnz_base + _block_scan(scan_sh, nnz, lane, warp)
        nnz_total = scan_sh[4]
        if rows == 1 and efcid < njmax_in:
          # every row so far is a one-row block below njmax, so the block index is the row index
          efc_jtdaj_adr_out[worldid, efcid] = efcid
          efc_jtdaj_nrow_out[worldid, efcid] = 1

          jntid_1 = eq_obj1id[eqid]
          data = eq_data[worldid % eq_data.shape[0], eqid]
          dofadr1 = jnt_dofadr[jntid_1]
          qposadr1 = jnt_qposadr[jntid_1]
          qpos0_id = worldid % qpos0.shape[0]
          dof_invweight0_id = worldid % dof_invweight0.shape[0]
          dofadr2 = int(-1)
          deriv_2 = float(0.0)
          if jntid_2 > -1:
            qposadr2 = jnt_qposadr[jntid_2]
            dofadr2 = jnt_dofadr[jntid_2]
            dif = qpos_in[worldid, qposadr2] - qpos0[qpos0_id, qposadr2]
            # Horner's method for polynomials
            rhs = data[0] + dif * (data[1] + dif * (data[2] + dif * (data[3] + dif * data[4])))
            deriv_2 = data[1] + dif * (2.0 * data[2] + dif * (3.0 * data[3] + dif * 4.0 * data[4]))
            pos = qpos_in[worldid, qposadr1] - qpos0[qpos0_id, qposadr1] - rhs
            Jqvel = qvel_in[worldid, dofadr1] - qvel_in[worldid, dofadr2] * deriv_2
            invweight = dof_invweight0[dof_invweight0_id, dofadr1] + dof_invweight0[dof_invweight0_id, dofadr2]
          else:
            pos = qpos_in[worldid, qposadr1] - qpos0[qpos0_id, qposadr1] - data[0]
            Jqvel = qvel_in[worldid, dofadr1]
            invweight = dof_invweight0[dof_invweight0_id, dofadr1]

          if rowadr + rownnz > njmax_nnz_in:
            # stock leaves this row half-written; publish an empty Jacobian row and flag the
            # overflow
            efc_J_rownnz_out[worldid, efcid] = 0
            efc_J_rowadr_out[worldid, efcid] = 0
            wp.atomic_or(overflow_out, worldid, OverflowType.NJMAX_NNZ)
          else:
            efc_J_rownnz_out[worldid, efcid] = rownnz
            efc_J_rowadr_out[worldid, efcid] = rowadr
            efc_J_colind_out[worldid, 0, rowadr] = dofadr1
            efc_J_out[worldid, 0, rowadr] = 1.0
            tree_a = dof_treeid[dofadr1]
            tree_b = int(-1)
            if jntid_2 > -1:
              efc_J_colind_out[worldid, 0, rowadr + 1] = dofadr2
              efc_J_out[worldid, 0, rowadr + 1] = -deriv_2
              tree_b = dof_treeid[dofadr2]
            # generic Jacobian-column scan of island._tree_edges
            first_tree = tree_a
            if first_tree < 0:
              first_tree = tree_b
              tree_b = -1
            if first_tree >= 0:
              if tree_b >= 0 and tree_b != first_tree:
                _mark_tree_edge(edge_sh, first_tree, tree_b)
              else:
                _mark_tree_edge(edge_sh, first_tree, -1)

          constraint._efc_row(
            opt_disableflags,
            worldid,
            timestep,
            efcid,
            pos,
            pos,
            invweight,
            eq_solref[worldid % eq_solref.shape[0], eqid],
            eq_solimp[worldid % eq_solimp.shape[0], eqid],
            0.0,
            Jqvel,
            0.0,
            ConstraintType.EQUALITY,
            eqid,
            efc_type_out,
            efc_id_out,
            efc_pos_out,
            efc_margin_out,
            efc_D_out,
            efc_vel_out,
            efc_aref_out,
            efc_frictionloss_out,
          )
        row_base += rows_total
        nnz_base += nnz_total
    ne = row_base

    # dof friction: one row per dof with frictionloss > 0
    if fric_on:
      for i0 in range(0, nv, BLOCK):
        dofid = i0 + tid
        rows = int(0)
        frictionloss = float(0.0)
        if dofid < nv:
          frictionloss = dof_frictionloss[worldid % dof_frictionloss.shape[0], dofid]
          if frictionloss > 0.0:
            rows = 1
        efcid = row_base + _block_scan(scan_sh, rows, lane, warp)
        rows_total = scan_sh[4]
        nnz = rows
        if efcid >= njmax_in:
          nnz = 0
        rowadr = nnz_base + _block_scan(scan_sh, nnz, lane, warp)
        nnz_total = scan_sh[4]
        if rows == 1 and efcid < njmax_in:
          efc_jtdaj_adr_out[worldid, efcid] = efcid
          efc_jtdaj_nrow_out[worldid, efcid] = 1
          if rowadr + 1 > njmax_nnz_in:
            efc_J_rownnz_out[worldid, efcid] = 0
            efc_J_rowadr_out[worldid, efcid] = 0
            wp.atomic_or(overflow_out, worldid, OverflowType.NJMAX_NNZ)
          else:
            efc_J_rownnz_out[worldid, efcid] = 1
            efc_J_rowadr_out[worldid, efcid] = rowadr
            efc_J_colind_out[worldid, 0, rowadr] = dofid
            efc_J_out[worldid, 0, rowadr] = 1.0
          _mark_tree_edge(edge_sh, dof_treeid[dofid], -1)
          constraint._efc_row(
            opt_disableflags,
            worldid,
            timestep,
            efcid,
            0.0,
            0.0,
            dof_invweight0[worldid % dof_invweight0.shape[0], dofid],
            dof_solref[worldid % dof_solref.shape[0], dofid],
            dof_solimp[worldid % dof_solimp.shape[0], dofid],
            0.0,
            qvel_in[worldid, dofid],
            frictionloss,
            ConstraintType.FRICTION_DOF,
            dofid,
            efc_type_out,
            efc_id_out,
            efc_pos_out,
            efc_margin_out,
            efc_D_out,
            efc_vel_out,
            efc_aref_out,
            efc_frictionloss_out,
          )
        row_base += rows_total
        nnz_base += nnz_total
    nf = row_base - ne

    # slide/hinge limits: one row per violated limit
    if lim_on:
      for i0 in range(0, nlimit, BLOCK):
        i = i0 + tid
        rows = int(0)
        jntid = int(-1)
        pos = float(0.0)
        dist_min = float(0.0)
        dist_max = float(0.0)
        jntmargin = float(0.0)
        if i < nlimit:
          jntid = jnt_limited_slide_hinge_adr[i]
          jntrange = jnt_range[worldid % jnt_range.shape[0], jntid]
          qpos = qpos_in[worldid, jnt_qposadr[jntid]]
          jntmargin = jnt_margin[worldid % jnt_margin.shape[0], jntid]
          dist_min = qpos - jntrange[0]
          dist_max = jntrange[1] - qpos
          pos = wp.min(dist_min, dist_max) - jntmargin
          if pos < 0.0:
            rows = 1
        efcid = row_base + _block_scan(scan_sh, rows, lane, warp)
        rows_total = scan_sh[4]
        nnz = rows
        if efcid >= njmax_in:
          nnz = 0
        rowadr = nnz_base + _block_scan(scan_sh, nnz, lane, warp)
        nnz_total = scan_sh[4]
        if rows == 1 and efcid < njmax_in:
          efc_jtdaj_adr_out[worldid, efcid] = efcid
          efc_jtdaj_nrow_out[worldid, efcid] = 1
          dofadr = jnt_dofadr[jntid]
          J = float(dist_min < dist_max) * 2.0 - 1.0
          if rowadr + 1 > njmax_nnz_in:
            efc_J_rownnz_out[worldid, efcid] = 0
            efc_J_rowadr_out[worldid, efcid] = 0
            wp.atomic_or(overflow_out, worldid, OverflowType.NJMAX_NNZ)
          else:
            efc_J_rownnz_out[worldid, efcid] = 1
            efc_J_rowadr_out[worldid, efcid] = rowadr
            efc_J_colind_out[worldid, 0, rowadr] = dofadr
            efc_J_out[worldid, 0, rowadr] = J
          _mark_tree_edge(edge_sh, dof_treeid[dofadr], -1)
          constraint._efc_row(
            opt_disableflags,
            worldid,
            timestep,
            efcid,
            pos,
            pos,
            dof_invweight0[worldid % dof_invweight0.shape[0], dofadr],
            jnt_solref[worldid % jnt_solref.shape[0], jntid],
            jnt_solimp[worldid % jnt_solimp.shape[0], jntid],
            jntmargin,
            J * qvel_in[worldid, dofadr],
            0.0,
            ConstraintType.LIMIT_JOINT,
            jntid,
            efc_type_out,
            efc_id_out,
            efc_pos_out,
            efc_margin_out,
            efc_D_out,
            efc_vel_out,
            efc_aref_out,
            efc_frictionloss_out,
          )
        row_base += rows_total
        nnz_base += nnz_total
    nl = row_base - ne - nf

    # pyramidal contacts of this world: ndim rows per active contact, one jtdaj block per contact.
    # Per chunk of BLOCK contacts, one thread per contact allocates rows/nonzeros/blocks and writes
    # the bookkeeping; the rows themselves are then distributed one per thread (a condim-6 contact
    # owns ten).
    nblock = wp.min(row_base, njmax_in)
    if con_on:
      ncon = world_con_count_in[worldid]
      cstart = world_con_start_in[worldid]
      for i0 in range(0, ncon, BLOCK):
        i = i0 + tid
        chunk_n = wp.min(BLOCK, ncon - i0)
        cid = int(-1)
        active = int(0)
        rows = int(0)
        nnz = int(0)
        ndim = int(0)
        rownnz = int(0)
        body1 = int(0)
        body2 = int(0)
        geom = wp.vec2i(0, 0)
        if i < ncon:
          cid = world_con_list_in[cstart + i]
          if contact_type_in[cid] & ContactType.CONSTRAINT:
            if contact_dist_in[cid] - contact_includemargin_in[cid] < 0.0:
              active = 1
              condim = contact_dim_in[cid]
              ndim = 1
              if condim > 1:
                ndim = 2 * (condim - 1)
              rows = ndim
              geom = contact_geom_in[cid]
              body1 = body_weldid[geom_bodyid[geom[0]]]
              body2 = body_weldid[geom_bodyid[geom[1]]]
              # count the merged ancestor chain excluding common dofs (constraint._efc_contact_init)
              da1 = body_dofadr[body1] + body_dofnum[body1] - 1
              da2 = body_dofadr[body2] + body_dofnum[body2] - 1
              while da1 >= 0 or da2 >= 0:
                da = wp.max(da1, da2)
                if da1 == da and da2 == da:
                  break
                if da1 == da:
                  da1 = dof_parentid[da1]
                if da2 == da:
                  da2 = dof_parentid[da2]
                rownnz += 1
              nnz = rownnz * ndim
        row_excl = _block_scan(scan_sh, rows, lane, warp)
        base = row_base + row_excl
        rows_total = scan_sh[4]
        blocks = int(0)
        if active == 1 and base < njmax_in:
          blocks = 1
        jgid = nblock + _block_scan(scan_sh, blocks, lane, warp)
        blocks_total = scan_sh[4]
        rowadr = nnz_base + _block_scan(scan_sh, nnz, lane, warp)
        nnz_total = scan_sh[4]
        nnz_ok = int(1)
        if rowadr + nnz > njmax_nnz_in:
          nnz_ok = 0

        if active == 1:
          for dim in range(ndim):
            efcid = base + dim
            if efcid >= njmax_in:
              contact_efc_address_out[cid, dim] = -1
            else:
              contact_efc_address_out[cid, dim] = efcid
              efc_id_out[worldid, efcid] = cid
              if nnz_ok == 1:
                efc_J_rowadr_out[worldid, efcid] = rowadr + dim * rownnz
                efc_J_rownnz_out[worldid, efcid] = rownnz
              else:
                efc_J_rowadr_out[worldid, efcid] = 0
                efc_J_rownnz_out[worldid, efcid] = 0
          if nnz_ok == 0:
            wp.atomic_or(overflow_out, worldid, OverflowType.NJMAX_NNZ)
          if blocks == 1:
            efc_jtdaj_adr_out[worldid, jgid] = base
            efc_jtdaj_nrow_out[worldid, jgid] = wp.min(ndim, njmax_in - base)
            _mark_tree_edge(edge_sh, body_treeid[geom_bodyid[geom[0]]], body_treeid[geom_bodyid[geom[1]]])
        else:
          cid = -1
        # stash the chunk's contacts for the row phase
        _st_chunk_i(c_cid_sh, tid, cid)
        _st_chunk_i(c_row_sh, tid, row_excl)
        _st_chunk_i(c_rowadr_sh, tid, rowadr)
        _st_chunk_i(c_rownnz_sh, tid, rownnz * nnz_ok)
        _st_chunk_i(c_body1_sh, tid, body1)
        _st_chunk_i(c_body2_sh, tid, body2)
        _sync()

        for r in range(tid, rows_total, BLOCK):
          efcid = row_base + r
          if efcid >= njmax_in:
            continue
          # owning contact: the last chunk entry whose row prefix is <= r (inactive contacts share
          # the prefix of their successor, so the search never lands on one)
          lo = int(0)
          hi = chunk_n - 1
          while lo < hi:
            mid = (lo + hi + 1) // 2
            if c_row_sh[mid] <= r:
              lo = mid
            else:
              hi = mid - 1
          cid = c_cid_sh[lo]
          dim = r - c_row_sh[lo]
          rownnz = c_rownnz_sh[lo]
          rowadr_dim = c_rowadr_sh[lo] + dim * rownnz
          body1 = c_body1_sh[lo]
          body2 = c_body2_sh[lo]

          condim = contact_dim_in[cid]
          geom = contact_geom_in[cid]
          includemargin = contact_includemargin_in[cid]
          pos = contact_dist_in[cid] - includemargin
          friction = contact_friction_in[cid]
          # row parameters (constraint._efc_contact_update): the inverse weight uses the geom bodies
          body_invweight0_id = worldid % body_invweight0.shape[0]
          invweight = (
            body_invweight0[body_invweight0_id, geom_bodyid[geom[0]]][0]
            + body_invweight0[body_invweight0_id, geom_bodyid[geom[1]]][0]
          )
          efc_type = int(ConstraintType.CONTACT_FRICTIONLESS)
          dimid2 = int(0)
          frii = float(0.0)
          if condim > 1:
            fri0 = friction[0]
            invweight = invweight + fri0 * fri0 * invweight
            invweight = invweight * 2.0 * fri0 * fri0 * impratio_invsqrt * impratio_invsqrt
            efc_type = int(ConstraintType.CONTACT_PYRAMIDAL)
            dimid2 = dim / 2 + 1
            frii = friction[dimid2 - 1]

          # pyramidal Jacobian row (constraint._efc_contact_jac_sparse); rownnz is 0 on nnz overflow
          Jqvel = float(0.0)
          if rownnz > 0:
            con_pos = contact_pos_in[cid]
            frame_0 = contact_frame_in[cid, 0]
            frame_i = wp.vec3(0.0)
            if condim > 1:
              if dimid2 < 3:
                frame_i = contact_frame_in[cid, dimid2]
              else:
                frame_i = contact_frame_in[cid, dimid2 - 3]
            da1 = body_dofadr[body1] + body_dofnum[body1] - 1
            da2 = body_dofadr[body2] + body_dofnum[body2] - 1
            da = wp.max(da1, da2)
            for k in range(rownnz):
              # common ancestors are excluded from rownnz, so one body owns this dof
              body = body2
              sign = float(1.0)
              if da1 == da:
                body = body1
                sign = -1.0
              cdof = cdof_in[worldid, da]
              cdof_ang = wp.spatial_top(cdof)
              offset = con_pos - subtree_com_in[worldid, body_rootid[body]]
              jacp_dif = (wp.spatial_bottom(cdof) + wp.cross(cdof_ang, offset)) * sign
              jacr_dif = cdof_ang * sign
              J = float(0.0)
              Ji = float(0.0)
              for xyz in range(3):
                J += frame_0[xyz] * jacp_dif[xyz]
                if condim > 1:
                  if dimid2 < 3:
                    Ji += frame_i[xyz] * jacp_dif[xyz]
                  else:
                    Ji += frame_i[xyz] * jacr_dif[xyz]
              if condim > 1:
                if dim % 2 == 0:
                  J += Ji * frii
                else:
                  J -= Ji * frii
              efc_J_colind_out[worldid, 0, rowadr_dim + k] = da
              efc_J_out[worldid, 0, rowadr_dim + k] = J
              Jqvel += J * qvel_in[worldid, da]
              if da1 == da:
                da1 = dof_parentid[da1]
              if da2 == da:
                da2 = dof_parentid[da2]
              da = wp.max(da1, da2)
          efc_Jqvel_out[worldid, efcid] = Jqvel
          constraint._efc_row(
            opt_disableflags,
            worldid,
            timestep,
            efcid,
            pos,
            pos,
            invweight,
            contact_solref_in[cid],
            contact_solimp_in[cid],
            includemargin,
            Jqvel,
            0.0,
            efc_type,
            cid,
            efc_type_out,
            efc_id_out,
            efc_pos_out,
            efc_margin_out,
            efc_D_out,
            efc_vel_out,
            efc_aref_out,
            efc_frictionloss_out,
          )
        _sync()
        row_base += rows_total
        nnz_base += nnz_total
        nblock += blocks_total

    if tid == 0:
      ne_out[worldid] = ne
      nf_out[worldid] = nf
      nl_out[worldid] = nl
      nefc_out[worldid] = row_base
      efc_jtdaj_nblock_out[worldid] = nblock
    _sync()

    # ----------------------------------------------------- M4: islands (island._flood_fill_bitsets)
    if tid == 0:
      for tree in range(ntree):
        island_nv_out[worldid, tree] = 0
      visited = wp.uint32(0)
      nisland = int(0)
      for root in range(ntree):
        root_bit = wp.uint32(1) << wp.uint32(root)
        if (visited & root_bit) != wp.uint32(0):
          continue
        if edge_sh[root] == wp.uint32(0):
          tree_island_out[worldid, root] = -1
          continue
        pending = root_bit
        island_nv = int(0)
        while pending != wp.uint32(0):
          tree = _lowest_set_bit(pending)
          tree_bit = wp.uint32(1) << wp.uint32(tree)
          pending &= ~tree_bit
          visited |= tree_bit
          neighbors = edge_sh[tree]
          tree_island_out[worldid, tree] = nisland
          island_nv += tree_dofnum[tree]
          pending |= neighbors & ~visited
        island_nv_out[worldid, nisland] = island_nv
        nisland += 1
      nisland_out[worldid] = nisland

  return kernel


@cache_kernel
def _forward_c_kernel(NB: int, NV: int, NU: int, NT: int, NSMALL: int, NBIGDOF: int, NBIG: int, REFRESH: bool):
  BLOCK = NV
  SMALL_SLOT = NSMALL * NSMALL + NSMALL
  BIG_SLOT = NBIGDOF * NBIGDOF + NBIGDOF
  FAC_SIZE = NT * SMALL_SLOT + NBIG * BIG_SLOT

  @wp.func_native(snippet="values.data(wp::tile_coord(index)) = value;")
  def _st_fac(values: wp.tile[float, FAC_SIZE], index: int, value: float): ...

  @wp.func_native(snippet="values.data(wp::tile_coord(index)) = value;")
  def _st_dof_f(values: wp.tile[float, NV], index: int, value: float): ...

  @wp.func_native(snippet="values.data(wp::tile_coord(index)) = value;")
  def _st_dof_sv(values: wp.tile[wp.spatial_vector, NV], index: int, value: wp.spatial_vector): ...

  @wp.func_native(snippet="values.data(wp::tile_coord(index)) = value;")
  def _st_act_f(values: wp.tile[float, NU], index: int, value: float): ...

  @wp.func_native(snippet="values.data(wp::tile_coord(index)) = value;")
  def _st_act_i(values: wp.tile[int, NU], index: int, value: int): ...

  @wp.func_native(snippet="values.data(wp::tile_coord(index)) = value;")
  def _st_body_sv(values: wp.tile[wp.spatial_vector, NB], index: int, value: wp.spatial_vector): ...

  @wp.func_native(snippet="values.data(wp::tile_coord(index)) = value;")
  def _st_body_v10(values: wp.tile[vec10, NB], index: int, value: vec10): ...

  @wp.kernel(module="unique", enable_backward=False, grid_stride=False, launch_bounds=(BLOCK,))
  def kernel(
    # Model:
    nbody: int,
    nv: int,
    nu: int,
    njnt: int,
    ntree: int,
    opt_timestep: wp.array[float],
    opt_disableflags: int,
    opt_gravity: wp.array[wp.vec3],
    opt_sleep_tolerance: wp.array[float],
    is_sparse: bool,
    warn_overflow: bool,
    qpos_spring: wp.array2d[float],
    body_parentid: wp.array[int],
    body_rootid: wp.array[int],
    body_mocapid: wp.array[int],
    body_treeid: wp.array[int],
    body_jntnum: wp.array[int],
    body_jntadr: wp.array[int],
    body_dofadr: wp.array[int],
    body_tree_offsets: wp.array[int],
    jnt_type: wp.array[int],
    jnt_qposadr: wp.array[int],
    jnt_dofadr: wp.array[int],
    jnt_stiffness: wp.array2d[float],
    jnt_stiffnesspoly: wp.array2d[wp.vec2],
    jnt_actgravcomp: wp.array[int],
    dof_bodyid: wp.array[int],
    dof_jntid: wp.array[int],
    dof_treeid: wp.array[int],
    dof_damping: wp.array2d[float],
    dof_dampingpoly: wp.array2d[wp.vec2],
    dof_length: wp.array[float],
    tree_dofadr: wp.array[int],
    tree_dofnum: wp.array[int],
    tree_sleep_policy: wp.array[int],
    M_rownnz: wp.array[int],
    M_rowadr: wp.array[int],
    M_colind: wp.array[int],
    qLD_block_adr: wp.array[int],
    actuator_trnid: wp.array[wp.vec2i],
    actuator_gear: wp.array2d[wp.spatial_vector],
    actuator_gaintype: wp.array[int],
    actuator_biastype: wp.array[int],
    actuator_gainprm: wp.array2d[vec10],
    actuator_biasprm: wp.array2d[vec10],
    actuator_forcelimited: wp.array[bool],
    actuator_forcerange: wp.array2d[wp.vec2],
    # Data in:
    nworld_in: int,
    naconmax_in: int,
    njmax_in: int,
    njmax_nnz_in: int,
    implicit_factor: int,
    nacon_in: wp.array[int],
    ncollision_in: wp.array[int],
    nefc_in: wp.array[int],
    efc_J_rownnz_in: wp.array2d[int],
    efc_J_rowadr_in: wp.array2d[int],
    M_in: wp.array2d[float],
    efc_Ma_in: wp.array2d[float],
    ctrl_in: wp.array2d[float],
    actuator_force_in: wp.array2d[float],
    qfrc_applied_in: wp.array2d[float],
    xfrc_applied_in: wp.array2d[wp.spatial_vector],
    nisland_in: wp.array[int],
    tree_island_in: wp.array2d[int],
    cdof_in: wp.array2d[wp.spatial_vector],
    cinert_in: wp.array2d[vec10],
    qfrc_gravcomp_in: wp.array2d[float],
    # Data out (state in/out):
    qpos_out: wp.array2d[float],
    qvel_out: wp.array2d[float],
    qacc_out: wp.array2d[float],
    time_out: wp.array[float],
    overflow_out: wp.array[int],
    qacc_warmstart_out: wp.array2d[float],
    tree_asleep_out: wp.array2d[int],
    tree_awake_out: wp.array2d[int],
    ntree_awake_out: wp.array[int],
    body_awake_out: wp.array2d[int],
    body_awake_ind_out: wp.array2d[int],
    nbody_awake_out: wp.array[int],
    dof_awake_ind_out: wp.array2d[int],
    nv_awake_out: wp.array[int],
    actuator_velocity_out: wp.array2d[float],
    cvel_out: wp.array2d[wp.spatial_vector],
    cdof_dot_out: wp.array2d[wp.spatial_vector],
    qfrc_spring_out: wp.array2d[float],
    qfrc_damper_out: wp.array2d[float],
    qfrc_passive_out: wp.array2d[float],
    cacc_out: wp.array2d[wp.spatial_vector],
    cfrc_int_out: wp.array2d[wp.spatial_vector],
    qfrc_bias_out: wp.array2d[float],
  ):
    worldid, tid = wp.tid()

    fac_sh = wp.tile_empty(shape=(FAC_SIZE,), dtype=float, storage="shared")
    qvel_sh = wp.tile_empty(shape=(NV,), dtype=float, storage="shared")
    qacc_sh = wp.tile_empty(shape=(NV,), dtype=float, storage="shared")
    act_vel_sh = wp.tile_empty(shape=(NU,), dtype=float, storage="shared")
    act_dof_sh = wp.tile_empty(shape=(NU,), dtype=int, storage="shared")
    tree_asleep_sh = wp.tile_empty(shape=(NT,), dtype=int, storage="shared")
    tree_island_sh = wp.tile_empty(shape=(NT,), dtype=int, storage="shared")
    tree_dofnum_sh = wp.tile_empty(shape=(NT,), dtype=int, storage="shared")
    tree_base_sh = wp.tile_empty(shape=(NT,), dtype=int, storage="shared")
    can_sleep_sh = wp.tile_empty(shape=(NT,), dtype=int, storage="shared")
    body_awake_sh = wp.tile_empty(shape=(NB,), dtype=int, storage="shared")
    misc_sh = wp.tile_empty(shape=(16,), dtype=int, storage="shared")

    lane = tid & 31
    warp = tid >> 5
    is_body = tid < nbody
    is_dof = tid < nv
    is_tree = tid < ntree
    timestep = opt_timestep[worldid % opt_timestep.shape[0]]
    dsbl_damper = (opt_disableflags & DisableBit.DAMPER) != 0

    # --------------------------------------------------- C0: loads and actuator velocity derivative
    t_n = int(0)
    t_start = int(0)
    t_compact = int(0)
    if is_tree:
      t_n = tree_dofnum[tid]
      t_start = tree_dofadr[tid]
      _st_tree32_i(tree_asleep_sh, tid, tree_asleep_out[worldid, tid])
      _st_tree32_i(tree_island_sh, tid, tree_island_in[worldid, tid])
      _st_tree32_i(tree_dofnum_sh, tid, t_n)
      _st_tree32_i(can_sleep_sh, tid, 1)
      if qLD_block_adr[t_start] == Q_LD_BLOCK_COMPACT:
        t_compact = 1
      base = tid * wp.static(SMALL_SLOT)
      if t_n > wp.static(NSMALL):
        nbig = int(0)
        for t in range(tid):
          if tree_dofnum[t] > wp.static(NSMALL):
            nbig += 1
        base = wp.static(NT * SMALL_SLOT) + nbig * wp.static(BIG_SLOT)
      _st_tree32_i(tree_base_sh, tid, base)
      if t_compact == 0 and implicit_factor != 0:
        for i in range(t_n * t_n):
          _st_fac(fac_sh, base + i, 0.0)
    d_body = int(0)
    d_tree = int(-1)
    d_qvel = float(0.0)
    d_qacc = float(0.0)
    d_qfrc_nz = int(0)
    if is_dof:
      d_body = dof_bodyid[tid]
      d_tree = dof_treeid[tid]
      d_qvel = qvel_out[worldid, tid]
      d_qacc = qacc_out[worldid, tid]
      if qfrc_applied_in[worldid, tid] != 0.0:
        d_qfrc_nz = 1
      _st_dof_f(qvel_sh, tid, d_qvel)
    b_xfrc_nz = int(0)
    b_tree = int(-1)
    if is_body:
      b_tree = body_treeid[tid]
      if not (xfrc_applied_in[worldid, tid] == wp.spatial_vector()):
        b_xfrc_nz = 1
    # derivative.deriv_smooth_vel: per-actuator velocity derivative of the AFFINE gain/bias terms,
    # skipped when the force sits at a forcerange bound; stored as moment^2 * vel for the diagonal
    for actid in range(tid, nu, BLOCK):
      gain = float(0.0)
      bias = float(0.0)
      if actuator_gaintype[actid] == GainType.AFFINE:
        gain = actuator_gainprm[worldid % actuator_gainprm.shape[0], actid][2]
      if actuator_biastype[actid] == BiasType.AFFINE:
        bias = actuator_biasprm[worldid % actuator_biasprm.shape[0], actid][2]
      vel = float(0.0)
      if not (bias == 0.0 and gain == 0.0):
        clamped = int(0)
        if actuator_forcelimited[actid]:
          force = actuator_force_in[worldid, actid]
          forcerange = actuator_forcerange[worldid % actuator_forcerange.shape[0], actid]
          if force <= forcerange[0] or force >= forcerange[1]:
            clamped = 1
        if clamped == 0:
          vel = bias
          if gain != 0.0:
            vel += gain * ctrl_in[worldid, actid]
      gear0 = actuator_gear[worldid % actuator_gear.shape[0], actid][0]
      _st_act_f(act_vel_sh, actid, gear0 * gear0 * vel)
      _st_act_i(act_dof_sh, actid, jnt_dofadr[actuator_trnid[actid][0]])
    _sync()

    # ------------------------------------------------------------ C1: implicitfast factor and solve
    # qDeriv = M - h * (actuator + damping velocity derivatives) touches the diagonal only for joint
    # transmissions; factor/solve as smooth.factor_solve_i with rhs Ma (support.mul_m from the
    # solver)
    if implicit_factor != 0:
      if is_dof:
        rowadr = M_rowadr[tid]
        rownnz = M_rownnz[tid]
        qderiv = float(0.0)
        for actid in range(nu):
          if act_dof_sh[actid] == tid:
            contrib = act_vel_sh[actid]
            if contrib != 0.0:
              qderiv += contrib
        if not dsbl_damper:
          damping = dof_damping[worldid % dof_damping.shape[0], tid]
          dpoly = dof_dampingpoly[worldid % dof_dampingpoly.shape[0], tid]
          qderiv -= util_misc._poly_force_deriv(damping, dpoly, d_qvel, 1)
        qderiv *= timestep
        diag = M_in[worldid, rowadr + rownnz - 1] - qderiv
        rhs = efc_Ma_in[worldid, tid]
        if qLD_block_adr[tid] == Q_LD_BLOCK_COMPACT:
          inverse = 1.0 / diag
          _st_dof_f(qacc_sh, tid, inverse * rhs)
        else:
          base = tree_base_sh[d_tree]
          start = tree_dofadr[d_tree]
          n = tree_dofnum_sh[d_tree]
          i = tid - start
          for k in range(rownnz - 1):
            col = M_colind[rowadr + k] - start
            _st_fac(fac_sh, base + i * n + col, M_in[worldid, rowadr + k])
          _st_fac(fac_sh, base + i * n + i, diag)
          _st_fac(fac_sh, base + n * n + i, rhs)
      _sync()
      if is_tree and t_compact == 0:
        n = t_n
        base = tree_base_sh[tid]
        xbase = base + n * n
        for i in range(n):
          diagonal_value = fac_sh[base + i * n + i]
          rhs_value = fac_sh[xbase + i]
          for k in range(i):
            factor = fac_sh[base + k * n + i]
            diagonal_value -= factor * factor
            rhs_value -= factor * fac_sh[xbase + k]
          diagonal_factor = wp.sqrt(diagonal_value)
          _st_fac(fac_sh, base + i * n + i, diagonal_factor)
          diagonal_inv = 1.0 / diagonal_factor
          _st_fac(fac_sh, xbase + i, rhs_value * diagonal_inv)
          for j in range(i + 1, n):
            value = fac_sh[base + j * n + i]
            for k in range(i):
              value -= fac_sh[base + k * n + i] * fac_sh[base + k * n + j]
            _st_fac(fac_sh, base + i * n + j, value * diagonal_inv)
        for reverse_i in range(n):
          i = n - 1 - reverse_i
          value = fac_sh[xbase + i]
          for k in range(i + 1, n):
            value -= fac_sh[base + i * n + k] * fac_sh[xbase + k]
          _st_fac(fac_sh, xbase + i, value / fac_sh[base + i * n + i])
        for i in range(n):
          _st_dof_f(qacc_sh, t_start + i, fac_sh[xbase + i])
    else:
      if is_dof:
        _st_dof_f(qacc_sh, tid, d_qacc)
    _sync()

    # --------------------------------------------------------- C2: advance velocity, position, time
    if is_dof:
      d_qvel = d_qvel + qacc_sh[tid] * timestep
      _st_dof_f(qvel_sh, tid, d_qvel)
      qacc_warmstart_out[worldid, tid] = d_qacc
    _sync()
    for jntid in range(tid, njnt, BLOCK):
      jnttype = jnt_type[jntid]
      qpos_adr = jnt_qposadr[jntid]
      dof_adr = jnt_dofadr[jntid]
      if jnttype == JointType.FREE:
        qpos_pos = wp.vec3(qpos_out[worldid, qpos_adr], qpos_out[worldid, qpos_adr + 1], qpos_out[worldid, qpos_adr + 2])
        qvel_lin = wp.vec3(qvel_sh[dof_adr], qvel_sh[dof_adr + 1], qvel_sh[dof_adr + 2])
        qpos_new = qpos_pos + timestep * qvel_lin
        qpos_quat = wp.quat(
          qpos_out[worldid, qpos_adr + 3],
          qpos_out[worldid, qpos_adr + 4],
          qpos_out[worldid, qpos_adr + 5],
          qpos_out[worldid, qpos_adr + 6],
        )
        qvel_ang = wp.vec3(qvel_sh[dof_adr + 3], qvel_sh[dof_adr + 4], qvel_sh[dof_adr + 5])
        qpos_quat_new = math.quat_integrate(qpos_quat, qvel_ang, timestep)
        qpos_out[worldid, qpos_adr + 0] = qpos_new[0]
        qpos_out[worldid, qpos_adr + 1] = qpos_new[1]
        qpos_out[worldid, qpos_adr + 2] = qpos_new[2]
        qpos_out[worldid, qpos_adr + 3] = qpos_quat_new[0]
        qpos_out[worldid, qpos_adr + 4] = qpos_quat_new[1]
        qpos_out[worldid, qpos_adr + 5] = qpos_quat_new[2]
        qpos_out[worldid, qpos_adr + 6] = qpos_quat_new[3]
      else:  # HINGE, SLIDE
        qpos_out[worldid, qpos_adr] = qpos_out[worldid, qpos_adr] + timestep * qvel_sh[dof_adr]
    if tid == 0:
      # forward._next_time: time and the capacity overflow flags
      time_out[worldid] = time_out[worldid] + timestep
      nefc = nefc_in[worldid]
      if nefc > njmax_in:
        if warn_overflow:
          wp.printf("nefc overflow - please increase njmax to %u\n", nefc)
        overflow_out[worldid] = overflow_out[worldid] | OverflowType.NEFC
      elif nefc > 0 and is_sparse:
        efcid = wp.min(nefc, njmax_in) - 1
        efc_nnz = efc_J_rowadr_in[worldid, efcid] + efc_J_rownnz_in[worldid, efcid]
        if efc_nnz > njmax_nnz_in:
          if warn_overflow:
            wp.printf("njmax_nnz overflow - please increase njmax_nnz to %u\n", efc_nnz)
          overflow_out[worldid] = overflow_out[worldid] | OverflowType.NJMAX_NNZ
      ncollision = ncollision_in[0]
      if ncollision > naconmax_in:
        if worldid == 0 and warn_overflow:
          nconmax = int(wp.ceil(float(ncollision) / float(nworld_in)))
          wp.printf("broadphase overflow - please increase nconmax to %u or naconmax to %u\n", nconmax, ncollision)
        overflow_out[worldid] = overflow_out[worldid] | OverflowType.BROADPHASE
      nacon = nacon_in[0]
      if nacon > naconmax_in:
        if worldid == 0 and warn_overflow:
          nconmax = int(wp.ceil(float(nacon) / float(nworld_in)))
          wp.printf("narrowphase overflow - please increase nconmax to %u or naconmax to %u\n", nconmax, nacon)
        overflow_out[worldid] = overflow_out[worldid] | OverflowType.NARROWPHASE

    # ---------------------------------------------------------------- C3: sleep.sleep
    # per-tree applied-force masks for _tree_can_sleep (ntree <= 32)
    xfrc_bits = wp.uint32(0)
    if is_body and b_xfrc_nz != 0 and b_tree >= 0:
      xfrc_bits = wp.uint32(1) << wp.uint32(b_tree)
    qfrc_bits = wp.uint32(0)
    if is_dof and d_qfrc_nz != 0 and d_tree >= 0:
      qfrc_bits = wp.uint32(1) << wp.uint32(d_tree)
    xfrc_bits = _warp_or(xfrc_bits)
    qfrc_bits = _warp_or(qfrc_bits)
    if lane == 0:
      _st_misc16_i(misc_sh, warp, int(xfrc_bits))
      _st_misc16_i(misc_sh, 4 + warp, int(qfrc_bits))
    _sync()

    # sweep: awake trees count down while quiet, otherwise reset to K_AWAKE_VAL
    if is_tree:
      as_val = tree_asleep_sh[tid]
      if as_val < 0:
        forced_bits = wp.uint32(0)
        for w in range(4):
          forced_bits |= wp.uint32(misc_sh[w]) | wp.uint32(misc_sh[4 + w])
        can_sleep = int(1)
        if tree_sleep_policy[tid] == SleepPolicy.AUTO_NEVER:
          can_sleep = 0
        elif ((forced_bits >> wp.uint32(tid)) & wp.uint32(1)) != wp.uint32(0):
          can_sleep = 0
        else:
          sleep_tolerance = opt_sleep_tolerance[worldid % opt_sleep_tolerance.shape[0]]
          for k in range(t_n):
            dof_idx = t_start + k
            v = qvel_sh[dof_idx]
            weight = dof_length[dof_idx]
            if sleep_tolerance > 0.0:
              if wp.abs(weight * v) >= sleep_tolerance:
                can_sleep = 0
            elif v != 0.0:
              can_sleep = 0
        if can_sleep == 1:
          if as_val < -1:
            _st_tree32_i(tree_asleep_sh, tid, as_val + 1)
        else:
          _st_tree32_i(tree_asleep_sh, tid, _K_AWAKE_VAL)
    _sync()

    # an island sleeps only when none of its trees is still counting down
    nisland = nisland_in[worldid]
    if is_tree:
      island_id = tree_island_sh[tid]
      if island_id >= 0 and island_id < nisland:
        if tree_asleep_sh[tid] < -1:
          _st_tree32_i(can_sleep_sh, island_id, 0)
    _sync()

    # build cycles (sleep._build_cycles): each tree writes only its own slot
    if is_tree:
      island_id = tree_island_sh[tid]
      sleeps = int(0)
      if island_id >= 0 and island_id < nisland:
        if can_sleep_sh[island_id] != 0:
          next_tree = int(tid)
          for offset in range(1, ntree):
            candidate = tid + offset
            if candidate >= ntree:
              candidate -= ntree
            if tree_island_sh[candidate] == island_id:
              next_tree = candidate
              break
          _st_tree32_i(tree_asleep_sh, tid, next_tree)
          sleeps = 1
      else:
        as_val = tree_asleep_sh[tid]
        if as_val == -1:
          _st_tree32_i(tree_asleep_sh, tid, tid)  # self-cycle
          sleeps = 1
        elif as_val >= 0:
          sleeps = 1
      if sleeps == 1:
        for k in range(t_n):
          _st_dof_f(qvel_sh, t_start + k, 0.0)
          qvel_out[worldid, t_start + k] = 0.0
          qacc_out[worldid, t_start + k] = 0.0
    _sync()
    if is_dof:
      # awake dofs keep the integrated velocity (sleeping trees were zeroed above)
      qvel_out[worldid, tid] = qvel_sh[tid]

    # --------------------------------------------------- C4: post-sleep velocity refresh (finalize)
    if wp.static(REFRESH):
      # fwd_velocity on the post-sleep qvel: cdof/cinert/subtree_com are the pre-integration values
      cdof_sh = wp.tile_empty(shape=(NV,), dtype=wp.spatial_vector, storage="shared")
      cvel_sh = wp.tile_empty(shape=(NB,), dtype=wp.spatial_vector, storage="shared")
      cacc_sh = wp.tile_empty(shape=(NB,), dtype=wp.spatial_vector, storage="shared")
      cinert_sh = wp.tile_empty(shape=(NB,), dtype=vec10, storage="shared")
      parent_sh = wp.tile_empty(shape=(NB,), dtype=int, storage="shared")
      nlevel = body_tree_offsets.shape[0] - 1
      gravity_enabled = (opt_disableflags & DisableBit.GRAVITY) == 0
      dsbl_spring = (opt_disableflags & DisableBit.SPRING) != 0
      gravity = opt_gravity[worldid % opt_gravity.shape[0]]

      b_parent = int(0)
      b_jntnum = int(0)
      b_dofadr = int(-1)
      j_type = int(-1)
      if is_body:
        b_parent = body_parentid[tid]
        b_jntnum = body_jntnum[tid]
        b_dofadr = body_dofadr[tid]
        if b_jntnum == 1:
          j_type = jnt_type[body_jntadr[tid]]
        _st_body64_i(parent_sh, tid, b_parent)
        _st_body_v10(cinert_sh, tid, cinert_in[worldid, tid])
      cdof = wp.spatial_vector()
      if is_dof:
        cdof = cdof_in[worldid, tid]
        _st_dof_sv(cdof_sh, tid, cdof)
      for actid in range(tid, nu, BLOCK):
        gear0 = actuator_gear[worldid % actuator_gear.shape[0], actid][0]
        actuator_velocity_out[worldid, actid] = gear0 * qvel_sh[act_dof_sh[actid]]
      if tid == 0:
        _st_body_sv(cvel_sh, 0, wp.spatial_vector())
        cacc0 = wp.spatial_vector()
        if gravity_enabled:
          cacc0 = wp.spatial_vector(wp.vec3(0.0), -gravity)
        _st_body_sv(cacc_sh, 0, cacc0)
      _sync()

      # body depth and subtree end from the depth-first numbering
      b_level = int(0)
      subtree_end = int(0)
      if is_body:
        p = int(tid)
        while p != 0:
          p = parent_sh[p]
          b_level += 1
        subtree_end = tid + 1
        while subtree_end < nbody:
          if parent_sh[subtree_end] < tid:
            break
          subtree_end += 1

      # com_vel and the velocity part of rne (level-synchronous)
      for level in range(1, nlevel):
        if is_body and b_level == level:
          cvel = cvel_sh[b_parent]
          cacc = cacc_sh[b_parent]
          if b_jntnum == 1:
            if j_type == JointType.FREE:
              for k in range(3):
                cvel += cdof_sh[b_dofadr + k] * qvel_sh[b_dofadr + k]
                cdof_dot_out[worldid, b_dofadr + k] = wp.spatial_vector()
              for k in range(3, 6):
                cdof_dot = math.motion_cross(cvel, cdof_sh[b_dofadr + k])
                cdof_dot_out[worldid, b_dofadr + k] = cdof_dot
                cacc += cdof_dot * qvel_sh[b_dofadr + k]
              for k in range(3, 6):
                cvel += cdof_sh[b_dofadr + k] * qvel_sh[b_dofadr + k]
            else:
              cdof_dot = math.motion_cross(cvel, cdof_sh[b_dofadr])
              cdof_dot_out[worldid, b_dofadr] = cdof_dot
              cacc += cdof_dot * qvel_sh[b_dofadr]
              cvel += cdof_sh[b_dofadr] * qvel_sh[b_dofadr]
          _st_body_sv(cvel_sh, tid, cvel)
          _st_body_sv(cacc_sh, tid, cacc)
        _sync()
      if is_body:
        cvel_out[worldid, tid] = cvel_sh[tid]
        cacc_out[worldid, tid] = cacc_sh[tid]

      # passive: springs on the advanced qpos, dampers on the post-sleep qvel, gravcomp unchanged
      if is_dof:
        qfrc_spring = float(0.0)
        qfrc_damper = float(0.0)
        d_jnt = dof_jntid[tid]
        if not (dsbl_spring and dsbl_damper):
          jnttype = jnt_type[d_jnt]
          jdof = jnt_dofadr[d_jnt]
          qposid = jnt_qposadr[d_jnt]
          stiffness = jnt_stiffness[worldid % jnt_stiffness.shape[0], d_jnt]
          spoly = jnt_stiffnesspoly[worldid % jnt_stiffnesspoly.shape[0], d_jnt]
          damping = dof_damping[worldid % dof_damping.shape[0], jdof]
          dpoly = dof_dampingpoly[worldid % dof_dampingpoly.shape[0], jdof]
          has_stiffness = (stiffness != 0.0 or spoly[0] != 0.0 or spoly[1] != 0.0) and not dsbl_spring
          has_damping = (damping != 0.0 or dpoly[0] != 0.0 or dpoly[1] != 0.0) and not dsbl_damper
          qpos_spring_id = worldid % qpos_spring.shape[0]
          v = qvel_sh[tid]
          if jnttype == JointType.FREE:
            if has_stiffness:
              k = tid - jdof
              if k < 3:
                dif = wp.vec3(
                  qpos_out[worldid, qposid + 0] - qpos_spring[qpos_spring_id, qposid + 0],
                  qpos_out[worldid, qposid + 1] - qpos_spring[qpos_spring_id, qposid + 1],
                  qpos_out[worldid, qposid + 2] - qpos_spring[qpos_spring_id, qposid + 2],
                )
                kf = util_misc._poly_force(stiffness, spoly, wp.length(dif), 0)
                qfrc_spring = -kf * dif[k]
              else:
                rot = wp.quat(
                  qpos_out[worldid, qposid + 3],
                  qpos_out[worldid, qposid + 4],
                  qpos_out[worldid, qposid + 5],
                  qpos_out[worldid, qposid + 6],
                )
                rot = wp.normalize(rot)
                ref = wp.quat(
                  qpos_spring[qpos_spring_id, qposid + 3],
                  qpos_spring[qpos_spring_id, qposid + 4],
                  qpos_spring[qpos_spring_id, qposid + 5],
                  qpos_spring[qpos_spring_id, qposid + 6],
                )
                dif = math.quat_sub(rot, ref)
                k_rot = util_misc._poly_force(stiffness, spoly, wp.length(dif), 0)
                qfrc_spring = -k_rot * dif[k - 3]
            if has_damping:
              qfrc_damper = -v * util_misc._poly_force(damping, dpoly, v, 1)
          else:  # SLIDE, HINGE
            if has_stiffness:
              fdif = qpos_out[worldid, qposid] - qpos_spring[qpos_spring_id, qposid]
              qfrc_spring = -fdif * util_misc._poly_force(stiffness, spoly, fdif, 0)
            if has_damping:
              qfrc_damper = -v * util_misc._poly_force(damping, dpoly, v, 1)
        qfrc_passive = qfrc_spring + qfrc_damper
        if gravity_enabled:
          if jnt_actgravcomp[d_jnt] == 0:
            qfrc_passive += qfrc_gravcomp_in[worldid, tid]
        qfrc_spring_out[worldid, tid] = qfrc_spring
        qfrc_damper_out[worldid, tid] = qfrc_damper
        qfrc_passive_out[worldid, tid] = qfrc_passive

      # rne: body forces, backward gather, qfrc_bias
      if is_body:
        frc = wp.spatial_vector()
        if tid > 0:
          cinert = cinert_sh[tid]
          cvel = cvel_sh[tid]
          frc = math.inert_vec(cinert, cacc_sh[tid])
          frc += math.motion_cross_force(cvel, math.inert_vec(cinert, cvel))
        _st_body_sv(cacc_sh, tid, frc)
      _sync()
      cfrc = wp.spatial_vector()
      if is_body:
        cfrc = cacc_sh[tid]
        for c in range(tid + 1, subtree_end):
          cfrc += cacc_sh[c]
      _sync()
      if is_body:
        _st_body_sv(cacc_sh, tid, cfrc)
        cfrc_int_out[worldid, tid] = cfrc
      _sync()
      if is_dof:
        qfrc_bias_out[worldid, tid] = wp.dot(cdof, cacc_sh[d_body])

    # ---------------------------------------------------------------- C5: sleep.update_sleep
    _publish_sleep_state(
      tree_asleep_sh,
      body_awake_sh,
      misc_sh,
      nbody,
      nv,
      ntree,
      body_rootid,
      body_mocapid,
      body_treeid,
      dof_bodyid,
      worldid,
      tid,
      tree_asleep_out,
      tree_awake_out,
      ntree_awake_out,
      body_awake_out,
      body_awake_ind_out,
      nbody_awake_out,
      dof_awake_ind_out,
      nv_awake_out,
    )

  return kernel


@event_scope
def forward_a(m: Model, d: Data):
  """Fused sleep wake/update, position, velocity and actuation stages (one CTA per world).

  Replaces ``sleep.wake`` + ``sleep.update_sleep``, ``smooth.kinematics``, ``smooth.com_pos``,
  ``smooth.crb``, ``smooth.transmission``, ``fwd_velocity`` (actuator velocity, ``com_vel``, passive
  forces, ``rne``) and ``fwd_actuation`` for models accepted by :func:`fused_world`.
  """
  wp.launch(
    _forward_a_kernel(NBODY_CAP, NV_CAP, NU_CAP, NTREE_CAP),
    dim=(d.nworld, NV_CAP),
    inputs=[
      m.nbody,
      m.nv,
      m.nu,
      m.ntree,
      m.ngeom,
      m.nsite,
      m.ngravcomp,
      m.opt.disableflags,
      m.opt.gravity,
      m.qpos0,
      m.qpos_spring,
      m.body_parentid,
      m.body_rootid,
      m.body_weldid,
      m.body_mocapid,
      m.body_jntnum,
      m.body_jntadr,
      m.body_dofnum,
      m.body_dofadr,
      m.body_treeid,
      m.body_pos,
      m.body_quat,
      m.body_ipos,
      m.body_iquat,
      m.body_mass,
      m.body_subtreemass,
      m.body_inertia,
      m.body_gravcomp,
      m.body_tree_offsets,
      m.gravcomp_bodyid,
      m.jnt_type,
      m.jnt_qposadr,
      m.jnt_dofadr,
      m.jnt_pos,
      m.jnt_axis,
      m.jnt_stiffness,
      m.jnt_stiffnesspoly,
      m.jnt_actgravcomp,
      m.jnt_actfrclimited,
      m.jnt_actfrcrange,
      m.dof_bodyid,
      m.dof_parentid,
      m.dof_jntid,
      m.dof_armature,
      m.dof_damping,
      m.dof_dampingpoly,
      m.tree_sleep_policy,
      m.M_rownnz,
      m.M_rowadr,
      m.geom_bodyid,
      m.geom_pos,
      m.geom_quat,
      m.site_bodyid,
      m.site_pos,
      m.site_quat,
      m.actuator_trnid,
      m.actuator_gear,
      m.actuator_gaintype,
      m.actuator_biastype,
      m.actuator_gainprm,
      m.actuator_biasprm,
      m.actuator_ctrllimited,
      m.actuator_ctrlrange,
      m.actuator_forcelimited,
      m.actuator_forcerange,
      d.qpos,
      d.qvel,
      d.qfrc_applied,
      d.xfrc_applied,
      d.mocap_pos,
      d.mocap_quat,
      d.ctrl,
    ],
    outputs=[
      d.tree_asleep,
      d.tree_awake,
      d.ntree_awake,
      d.body_awake,
      d.body_awake_ind,
      d.nbody_awake,
      d.dof_awake_ind,
      d.nv_awake,
      d.xpos,
      d.xquat,
      d.xmat,
      d.xipos,
      d.ximat,
      d.xanchor,
      d.xaxis,
      d.geom_xpos,
      d.geom_xmat,
      d.site_xpos,
      d.site_xmat,
      d.subtree_com,
      d.cinert,
      d.cdof,
      d.crb,
      d.M,
      d.actuator_length,
      d.moment_rownnz,
      d.moment_rowadr,
      d.moment_colind,
      d.actuator_moment,
      d.actuator_velocity,
      d.cvel,
      d.cdof_dot,
      d.qfrc_spring,
      d.qfrc_damper,
      d.qfrc_gravcomp,
      d.qfrc_adhesion,
      d.qfrc_passive,
      d.cacc,
      d.cfrc_int,
      d.qfrc_bias,
      d.actuator_force,
      d.qfrc_actuator,
    ],
    block_dim=NV_CAP,
  )


@event_scope
def forward_b(m: Model, d: Data):
  """Fused acceleration stage as one CTA per world.

  ``qfrc_smooth``, ``xfrc_accumulate``, the per-tree factor/solve with the sleeping-tree freeze and
  the active-DOF compaction maps (``island.update_active_dofs``).
  """
  wp.launch(
    _forward_b_kernel(NBODY_CAP, NV_CAP, NTREE_CAP, NVTREE_SMALL, NVTREE_CAP, NBIG_CAP),
    dim=(d.nworld, NV_CAP),
    inputs=[
      m.nbody,
      m.nv,
      m.ntree,
      m.body_parentid,
      m.body_rootid,
      m.body_treeid,
      m.dof_bodyid,
      m.dof_treeid,
      m.tree_dofadr,
      m.tree_dofnum,
      m.M_rownnz,
      m.M_rowadr,
      m.M_colind,
      m.qLD_block_adr,
      d.nvmax,
      d.nvmax_pad,
      types.TILE_SIZE_JTDAJ_DENSE,
      bool(m.opt.warn_overflow),
      d.tree_awake,
      d.tree_island,
      d.island_nv,
      d.qfrc_passive,
      d.qfrc_bias,
      d.qfrc_actuator,
      d.qfrc_applied,
      d.xfrc_applied,
      d.xipos,
      d.subtree_com,
      d.cdof,
      d.M,
    ],
    outputs=[
      d.qfrc_smooth,
      d.qLD,
      d.qLDiagInv,
      d.qacc_smooth,
      d.dof_cdof,
      d.cdof_dof,
      d.ncdof,
      d.nsingleton6,
      d.overflow,
    ],
    block_dim=NV_CAP,
  )


def _group_contacts_by_world(m: Model, d: Data):
  """Counting sort of the live contacts by world; returns (start, count, list) scratch arrays."""
  count_fill = wp.zeros((2 * d.nworld,), dtype=int)
  count = count_fill[: d.nworld]
  fill = count_fill[d.nworld :]
  start = wp.empty((d.nworld,), dtype=int)
  con_list = wp.empty((max(d.naconmax, 1),), dtype=int)
  if m.opt.disableflags & DisableBit.CONTACT:
    return start, count, con_list
  threads = min(_GROUP_THREADS, max(d.naconmax, 1))
  wp.launch(_world_contact_count, dim=threads, inputs=[d.nacon, d.contact.worldid, threads], outputs=[count])
  wp.launch(
    _world_contact_scatter,
    dim=threads,
    inputs=[d.nworld, d.nacon, d.contact.worldid, count, threads],
    outputs=[fill, start, con_list],
    block_dim=NV_CAP,
  )
  return start, count, con_list


@event_scope
def forward_m(m: Model, d: Data):
  """Fused constraint assembly and sleep/island bookkeeping between ``forward_a`` and ``forward_b``.

  Replaces ``constraint.make_constraint`` (JOINT equalities, dof friction, slide/hinge limits and
  pyramidal contact rows), ``sleep.wake_equality``, ``sleep.update_sleep`` and ``island.island`` for
  models accepted by :func:`fused_world`; two small grouping launches sort the contacts by world
  first.
  """
  start, count, con_list = _group_contacts_by_world(m, d)
  _launch_forward_m(m, d, start, count, con_list)


def _launch_forward_m(m: Model, d: Data, start: wp.array, count: wp.array, con_list: wp.array):
  contact_frame_2d = wp.array(
    ptr=d.contact.frame.ptr, dtype=wp.vec3, shape=(d.naconmax, 3), device=d.contact.frame.device, copy=False
  )
  wp.launch(
    _forward_m_kernel(NBODY_CAP, NV_CAP, NTREE_CAP),
    dim=(d.nworld, NV_CAP),
    inputs=[
      m.nv,
      m.nbody,
      m.ntree,
      m.neq,
      m.eq_jnt_adr.size,
      m.jnt_limited_slide_hinge_adr.size,
      m.opt.timestep,
      m.opt.disableflags,
      m.opt.impratio_invsqrt,
      m.qpos0,
      m.body_rootid,
      m.body_weldid,
      m.body_mocapid,
      m.body_treeid,
      m.body_dofnum,
      m.body_dofadr,
      m.body_invweight0,
      m.jnt_qposadr,
      m.jnt_dofadr,
      m.jnt_bodyid,
      m.jnt_solref,
      m.jnt_solimp,
      m.jnt_range,
      m.jnt_margin,
      m.jnt_limited_slide_hinge_adr,
      m.dof_bodyid,
      m.dof_parentid,
      m.dof_treeid,
      m.dof_solref,
      m.dof_solimp,
      m.dof_frictionloss,
      m.dof_invweight0,
      m.tree_dofnum,
      m.geom_bodyid,
      m.eq_obj1id,
      m.eq_obj2id,
      m.eq_solref,
      m.eq_solimp,
      m.eq_data,
      m.eq_jnt_adr,
      d.njmax,
      d.njmax_nnz,
      d.qpos,
      d.qvel,
      d.eq_active,
      d.tree_awake,
      d.subtree_com,
      d.cdof,
      d.contact.dist,
      d.contact.dim,
      d.contact.includemargin,
      d.contact.geom,
      d.contact.type,
      d.contact.pos,
      contact_frame_2d,
      d.contact.friction,
      d.contact.solref,
      d.contact.solimp,
      start,
      count,
      con_list,
    ],
    outputs=[
      d.tree_asleep,
      d.tree_awake,
      d.ntree_awake,
      d.body_awake,
      d.body_awake_ind,
      d.nbody_awake,
      d.dof_awake_ind,
      d.nv_awake,
      d.ne,
      d.nf,
      d.nl,
      d.nefc,
      d.efc.type,
      d.efc.id,
      d.efc.jtdaj_adr,
      d.efc.jtdaj_nrow,
      d.efc.jtdaj_nblock,
      d.efc.J_rownnz,
      d.efc.J_rowadr,
      d.efc.J_colind,
      d.efc.J,
      d.efc.pos,
      d.efc.margin,
      d.efc.D,
      d.efc.vel,
      d.efc.aref,
      d.efc.frictionloss,
      d.efc.Jqvel,
      d.contact.efc_address,
      d.overflow,
      d.nisland,
      d.tree_island,
      d.island_nv,
    ],
    block_dim=NV_CAP,
  )


def implicit_factor(m: Model) -> bool:
  """Whether implicitfast factors ``M - h*D`` (forward._implicit: actuation, spring or damper)."""
  return bool(~(m.opt.disableflags | ~(DisableBit.ACTUATION | DisableBit.SPRING | DisableBit.DAMPER)))


@event_scope
def forward_c(m: Model, d: Data, *, finalize: bool):
  """Fused implicitfast integration and sleep bookkeeping after ``solver.solve``, one CTA per world.

  Replaces ``forward._implicit`` for IMPLICITFAST: ``derivative.deriv_smooth_vel`` +
  ``factor_solve_i`` with rhs ``efc.Ma``, ``_advance`` (velocity, position, time and overflow flags,
  warmstart copy), ``sleep.sleep``, the post-sleep ``fwd_velocity`` refresh (``finalize`` only; the
  predicate excludes the spatial equalities and callbacks that would otherwise require it) and
  ``sleep.update_sleep``.
  """
  wp.launch(
    _forward_c_kernel(NBODY_CAP, NV_CAP, NU_CAP, NTREE_CAP, NVTREE_SMALL, NVTREE_CAP, NBIG_CAP, bool(finalize)),
    dim=(d.nworld, NV_CAP),
    inputs=[
      m.nbody,
      m.nv,
      m.nu,
      m.njnt,
      m.ntree,
      m.opt.timestep,
      m.opt.disableflags,
      m.opt.gravity,
      m.opt.sleep_tolerance,
      bool(m.is_sparse),
      bool(m.opt.warn_overflow),
      m.qpos_spring,
      m.body_parentid,
      m.body_rootid,
      m.body_mocapid,
      m.body_treeid,
      m.body_jntnum,
      m.body_jntadr,
      m.body_dofadr,
      m.body_tree_offsets,
      m.jnt_type,
      m.jnt_qposadr,
      m.jnt_dofadr,
      m.jnt_stiffness,
      m.jnt_stiffnesspoly,
      m.jnt_actgravcomp,
      m.dof_bodyid,
      m.dof_jntid,
      m.dof_treeid,
      m.dof_damping,
      m.dof_dampingpoly,
      m.dof_length,
      m.tree_dofadr,
      m.tree_dofnum,
      m.tree_sleep_policy,
      m.M_rownnz,
      m.M_rowadr,
      m.M_colind,
      m.qLD_block_adr,
      m.actuator_trnid,
      m.actuator_gear,
      m.actuator_gaintype,
      m.actuator_biastype,
      m.actuator_gainprm,
      m.actuator_biasprm,
      m.actuator_forcelimited,
      m.actuator_forcerange,
      d.nworld,
      d.naconmax,
      d.njmax,
      d.njmax_nnz,
      int(implicit_factor(m)),
      d.nacon,
      d.ncollision,
      d.nefc,
      d.efc.J_rownnz,
      d.efc.J_rowadr,
      d.M,
      d.efc.Ma,
      d.ctrl,
      d.actuator_force,
      d.qfrc_applied,
      d.xfrc_applied,
      d.nisland,
      d.tree_island,
      d.cdof,
      d.cinert,
      d.qfrc_gravcomp,
    ],
    outputs=[
      d.qpos,
      d.qvel,
      d.qacc,
      d.time,
      d.overflow,
      d.qacc_warmstart,
      d.tree_asleep,
      d.tree_awake,
      d.ntree_awake,
      d.body_awake,
      d.body_awake_ind,
      d.nbody_awake,
      d.dof_awake_ind,
      d.nv_awake,
      d.actuator_velocity,
      d.cvel,
      d.cdof_dot,
      d.qfrc_spring,
      d.qfrc_damper,
      d.qfrc_passive,
      d.cacc,
      d.cfrc_int,
      d.qfrc_bias,
    ],
    block_dim=NV_CAP,
  )
