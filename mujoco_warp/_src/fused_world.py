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

"""Per-world CTA fusion of the smooth forward dynamics for small sleeping models.

Two kernels, one 128-thread CTA per world, replace the launch sequence of ``forward()`` up to
the constraint solver for models that satisfy :func:`fused_world`:

* ``forward_a``: ``sleep.wake`` + ``sleep.update_sleep``, kinematics, ``com_pos``, ``crb`` and the
  sparse inertia ``M``, joint transmission, ``com_vel``, passive forces (springs, dampers, gravity
  compensation), RNE bias forces and the position/velocity servo actuator forces.
* ``forward_b``: ``qfrc_smooth`` (with the sleeping-tree freeze), ``xfrc_accumulate``, the per-tree
  factor/solve for ``qacc_smooth`` (``qLD``/``qLDiagInv``) and the active-DOF compaction maps that
  the compact constraint solver consumes.

``make_constraint``, ``wake_equality``, ``update_sleep`` and ``island`` stay native between the two
kernels; the constraint solver and the integrator are untouched. Every canonical ``Data`` field
written by the replaced kernels is published so that downstream consumers see identical state.

Per-body and per-dof state lives in shared memory. Tree traversals use the depth-first body numbering
of MuJoCo (a subtree is a contiguous id range), so backward accumulations are gather sums without
atomics, and forward passes are level-synchronous with all bodies of a depth processed in parallel.
Bodies (with their single joint), dofs, actuators and trees are owned by one thread each; the block
dimension equals the dof capacity.
"""

import os

import numpy as np
import warp as wp

from mujoco_warp._src import math
from mujoco_warp._src import types
from mujoco_warp._src import util_misc
from mujoco_warp._src.types import Q_LD_BLOCK_COMPACT
from mujoco_warp._src.types import Q_LD_BLOCK_SPARSE
from mujoco_warp._src.types import BiasType
from mujoco_warp._src.types import ConeType
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
from mujoco_warp._src.types import vec10
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


def _static_eligible(m: Model) -> bool:
  """Structural (model-only) part of the eligibility predicate; evaluated once per model."""
  if m.nv == 0 or m.ntree == 0 or m.nbody < 2:
    return False
  if m.nv > NV_CAP or m.nbody > NBODY_CAP or m.nu > NU_CAP or m.ntree > NTREE_CAP:
    return False
  if m.ntendon or m.nflex or m.na or m.nhistory or m.nacttrnbody:
    return False
  if m.has_fluid or m.flg_adhesion:
    return False
  if not m.is_sparse:
    return False

  jnt_type = m.jnt_type.numpy()
  if not np.isin(jnt_type, (JointType.FREE, JointType.HINGE, JointType.SLIDE)).all():
    return False
  if m.body_jntnum.numpy().max() > 1:
    return False
  if m.neq and not (m.eq_type.numpy() == EqType.JOINT).all():
    return False

  if m.nu:
    trntype = m.actuator_trntype.numpy()
    if not np.isin(trntype, (TrnType.JOINT, TrnType.JOINTINPARENT)).all():
      return False
    trn_jnt = m.actuator_trnid.numpy()[:, 0]
    if not np.isin(jnt_type[trn_jnt], (JointType.HINGE, JointType.SLIDE)).all():
      return False
    if not (m.actuator_dyntype.numpy() == DynType.NONE).all():
      return False
    if not np.isin(m.actuator_gaintype.numpy(), (GainType.FIXED, GainType.AFFINE)).all():
      return False
    if not np.isin(m.actuator_biastype.numpy(), (BiasType.NONE, BiasType.AFFINE)).all():
      return False
    if m.actuator_actearly.numpy().any():
      return False
    if m.nJmom < m.nu:
      return False

  tree_dofnum = m.tree_dofnum.numpy()
  if tree_dofnum.max() > NVTREE_CAP or (tree_dofnum > NVTREE_SMALL).sum() > NBIG_CAP:
    return False
  if (m.qLD_block_adr.numpy() == Q_LD_BLOCK_SPARSE).any():
    return False

  return _dfs_contiguous(m.body_parentid.numpy())


def fused_world(m: Model, d: Data) -> bool:
  """Return whether the fused per-world forward kernels apply to ``(m, d)``.

  The structural part is cached on the model; option flags and runtime toggles (such as
  ``sensor_rne_postconstraint``, which Newton flips at runtime) are re-evaluated on every call.
  """
  if not enabled:
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

  cached = m.__dict__.get("_fused_world_static")
  if cached is None:
    cached = _static_eligible(m)
    m.__dict__["_fused_world_static"] = cached
  return cached


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

    # ---------------------------------------------------------------- P0: prefetch (one latency round)
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

    # per-tree wake masks (ntree <= 32): applied Cartesian force, applied generalized force, velocity
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

    # depth-first numbering: the subtree of tid is [tid, subtree_end); body level from the parent walk
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

    # ---------------------------------------------------------------- P2: kinematics (level-synchronous)
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

    # ---------------------------------------------------------------- P2b: body frames, joints, geoms, sites
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

    # ---------------------------------------------------------------- P5: transmission, actuator velocity
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

    # ---------------------------------------------------------------- P6: com_vel and cacc (level-synchronous)
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

        # gravity compensation: bodies in this dof's subtree (support.jac_dof via body_isdofancestor)
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

    # ---------------------------------------------------------------- Q0: loads, map reset, scratch slots
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

    # ---------------------------------------------------------------- Q1: qfrc_smooth + xfrc_accumulate
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

    # ---------------------------------------------------------------- Q2: per-tree dense factor and solve
    if is_tree and t_compact == 0:
      n = t_n
      base = tree_base_sh[tid]
      xbase = base + n * n
      awake = tree_awake_sh[tid]
      factor_adr = qLD_block_adr[t_start]

      # upper Cholesky M = U^T U with the solve interleaved (smooth._small_cholesky_factorize_solve_block)
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

    # ---------------------------------------------------------------- Q3: compaction layout (island._compact_dof_layout)
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

    # ---------------------------------------------------------------- Q3b: compaction maps (island._map_compact_dofs)
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
  """Fused acceleration stage: ``qfrc_smooth``, ``xfrc_accumulate``, per-tree factor/solve, freeze and
  the active-DOF compaction maps (``island.update_active_dofs``) as one CTA per world."""
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
