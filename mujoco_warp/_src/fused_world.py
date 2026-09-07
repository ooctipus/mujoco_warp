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
  ``sleep.update_sleep`` and the ``island`` bitset flood fill. One small bucket launch (contact ids
  per world) precedes it because externally supplied contacts carry a global id order.
* ``forward_b``: ``qfrc_smooth`` (with the sleeping-tree freeze), ``xfrc_accumulate``, the per-tree
  factor/solve for ``qacc_smooth`` (``qLD``/``qLDiagInv``) and the active-DOF compaction maps that
  the compact constraint solver consumes.
* ``forward_c`` (after ``solver.solve``): the implicitfast factor ``M - h*D`` and solve with rhs
  ``efc.Ma``, velocity/position/time advance with the overflow flags, the warmstart copy,
  ``sleep.sleep`` (countdown, island can-sleep, cycles with zeroed ``qvel``/``qacc``), the
  post-sleep ``fwd_velocity`` refresh when the step finalizes and ``sleep.update_sleep``.

Every canonical ``Data`` field written by the replaced kernels is published so that downstream
consumers see identical state, unless ``Option.fused_world_publish_derived`` is False (an opt-in
that skips :data:`DERIVED_FIELDS`, which no fused stage reads; see :func:`publish_derived`).
Integer and sleep/island state is reproduced exactly; floating point
differs from the stock launches only through summation and factorization order (scalar instead of
tile Cholesky for the dense inertia blocks: fp32 rounding, a few ulps in ``qLD``/``qacc_smooth``).
Row order within a constraint family and the ``body_awake_ind``/``dof_awake_ind`` order are
deterministic given the contact list order (stock: atomic allocation order); the contact list order
itself follows the grouping atomics.

Per-body and per-dof state lives in shared memory. Tree traversals use the depth-first body
numbering of MuJoCo (a subtree is a contiguous id range), so backward accumulations are gather sums
without atomics, and the forward passes (kinematics, com_vel, cacc) are ancestor walks: each body
composes or sums the per-body terms of its chain from shared memory after one barrier, instead of
one barrier per tree level. Bodies (with their single joint), dofs, actuators and trees are owned by
one thread each; the block dimension equals the dof capacity.

``forward_a`` also runs over a subset of worlds (``world_ids``): CTA ``slot`` serves world
``world_ids[slot]``, and an optional device count lets the caller size the launch without a host
sync (CTAs at or beyond the count exit before touching any state). ``forward.forward_worlds`` uses
this to rebuild the derived data of reset worlds only.

Warp pitfalls handled here: tile element assignment embeds a block barrier, so shared stores in
thread-divergent code use single-line native snippets; multi-line native snippets live at module
scope.
"""

import dataclasses

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


@wp.func_native(snippet="WP_TILE_SYNC();")
def _sync():
  pass


@wp.func_native(snippet="__syncwarp();")
def _syncwarp():
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
for (int offset = 16; offset > 0; offset >>= 1) value += __shfl_xor_sync(0xffffffffu, value, offset);
return value;
#else
return value;
#endif
"""
)
def _warp_sum(value: float) -> float: ...


@wp.func_native(
  snippet="""
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
return __reduce_max_sync(0xffffffffu, value);
#elif defined(__CUDA_ARCH__)
for (int offset = 16; offset > 0; offset >>= 1) value = max(value, __shfl_xor_sync(0xffffffffu, value, offset));
return value;
#else
return value;
#endif
"""
)
def _warp_max(value: int) -> int: ...


# Segmented sum of up to ten lane values towards the first lane of each segment (lanes with equal
# ``seg``; segments are contiguous and never straddle a warp). ``ndim`` must be warp-uniform: it
# bounds the components reduced. All 32 lanes must participate (idle lanes pass seg = -1).
@wp.func_native(
  snippet="""
#if defined(__CUDA_ARCH__)
const int lane = threadIdx.x & 31;
for (int offset = 1; offset < 32; offset <<= 1) {
  const int other_seg = __shfl_down_sync(0xffffffffu, seg, offset);
  const bool take = (lane + offset < 32) && (other_seg == seg);
#pragma unroll
  for (int dim = 0; dim < 10; ++dim) {
    if (dim < ndim) {
      const float other = __shfl_down_sync(0xffffffffu, values.c[dim], offset);
      if (take) values.c[dim] += other;
    }
  }
}
#endif
return values;
"""
)
def _segment_sum_head10(values: vec10, seg: int, ndim: int) -> vec10: ...


# Vectorized publishes of the array-of-structures Data fields. Warp stores a struct element as
# scalar 4-byte stores, so a warp writing 24/36/40-byte elements touches 6-10x more 32-byte sectors
# than the data needs; these snippets write 8- or 16-byte words when the element address is aligned
# (contiguous Data arrays) and fall back to the scalar stores otherwise.
@wp.func_native(
  snippet="""
float* p = reinterpret_cast<float*>(&wp::index(arr, i, j));
if ((reinterpret_cast<size_t>(p) & 7) == 0) {
  float2* q = reinterpret_cast<float2*>(p);
  q[0] = make_float2(value.c[0], value.c[1]);
  q[1] = make_float2(value.c[2], value.c[3]);
  q[2] = make_float2(value.c[4], value.c[5]);
} else {
  for (int k = 0; k < 6; ++k) p[k] = value.c[k];
}
"""
)
def _st_sv(arr: wp.array2d[wp.spatial_vector], i: int, j: int, value: wp.spatial_vector): ...


@wp.func_native(
  snippet="""
float* p = reinterpret_cast<float*>(&wp::index(arr, i, j));
if ((reinterpret_cast<size_t>(p) & 7) == 0) {
  float2* q = reinterpret_cast<float2*>(p);
  q[0] = make_float2(value.c[0], value.c[1]);
  q[1] = make_float2(value.c[2], value.c[3]);
  q[2] = make_float2(value.c[4], value.c[5]);
  q[3] = make_float2(value.c[6], value.c[7]);
  q[4] = make_float2(value.c[8], value.c[9]);
} else {
  for (int k = 0; k < 10; ++k) p[k] = value.c[k];
}
"""
)
def _st_v10(arr: wp.array2d[vec10], i: int, j: int, value: vec10): ...


@wp.func_native(
  snippet="""
float* p = reinterpret_cast<float*>(&wp::index(arr, i, j));
if ((reinterpret_cast<size_t>(p) & 15) == 0) {
  *reinterpret_cast<float4*>(p) = make_float4(value.x, value.y, value.z, value.w);
} else {
  p[0] = value.x; p[1] = value.y; p[2] = value.z; p[3] = value.w;
}
"""
)
def _st_q(arr: wp.array2d[wp.quat], i: int, j: int, value: wp.quat): ...


@wp.func_native(
  snippet="""
float* p = reinterpret_cast<float*>(&wp::index(arr, i, j));
const float* v = reinterpret_cast<const float*>(&value);
if ((reinterpret_cast<size_t>(p) & 7) == 0) {
  float2* q = reinterpret_cast<float2*>(p);
  q[0] = make_float2(v[0], v[1]); q[1] = make_float2(v[2], v[3]);
  q[2] = make_float2(v[4], v[5]); q[3] = make_float2(v[6], v[7]);
  p[8] = v[8];
} else {
  p[0] = v[0];
  float2* q = reinterpret_cast<float2*>(p + 1);
  q[0] = make_float2(v[1], v[2]); q[1] = make_float2(v[3], v[4]);
  q[2] = make_float2(v[5], v[6]); q[3] = make_float2(v[7], v[8]);
}
"""
)
def _st_m33(arr: wp.array2d[wp.mat33], i: int, j: int, value: wp.mat33): ...


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


@wp.func_native(snippet="values.data(wp::tile_coord(index)) = value;")
def _st_chunk_f(values: wp.tile[float, NV_CAP], index: int, value: float): ...


# dense per-tree scratch of forward_b / forward_c: n*n factor, then n solution entries per tree
_SMALL_SLOT = NVTREE_SMALL * NVTREE_SMALL + NVTREE_SMALL
_BIG_SLOT = NVTREE_CAP * NVTREE_CAP + NVTREE_CAP
_FAC_SIZE = NTREE_CAP * _SMALL_SLOT + NBIG_CAP * _BIG_SLOT


@wp.func_native(snippet="values.data(wp::tile_coord(index)) = value;")
def _st_fac_sh(values: wp.tile[float, _FAC_SIZE], index: int, value: float): ...


@wp.func
def _dense_factor_solve(fac_sh: wp.tile[float, _FAC_SIZE], active: bool, base: int, n: int, j: int, nmax: int):
  """Upper Cholesky ``A = U^T U`` of an ``n x n`` block and the solve of its right-hand side.

  The block starts at ``base`` with ``A`` in the lower triangle and the right-hand side at
  ``base + n * n``; ``U`` is written to the upper triangle and the solution replaces the right-hand
  side. Each calling thread owns column ``j`` of its block (``active`` threads only) and accumulates
  in the order of the single-thread ``smooth._small_cholesky_factorize_solve_block``, so the results
  are bitwise those of the serial factorization. Contains ``2 * nmax`` block barriers: every thread
  must call it with the same ``nmax`` (>= n of every block).
  """
  xbase = base + n * n
  for i in range(nmax):
    if active and i < n and j >= i:
      diagonal_value = fac_sh[base + i * n + i]
      for k in range(i):
        factor = fac_sh[base + k * n + i]
        diagonal_value -= factor * factor
      diagonal_factor = wp.sqrt(diagonal_value)
      diagonal_inv = 1.0 / diagonal_factor
      if j == i:
        rhs_value = fac_sh[xbase + i]
        for k in range(i):
          rhs_value -= fac_sh[base + k * n + i] * fac_sh[xbase + k]
        _st_fac_sh(fac_sh, base + i * n + i, diagonal_factor)
        _st_fac_sh(fac_sh, xbase + i, rhs_value * diagonal_inv)
      else:
        value = fac_sh[base + j * n + i]
        for k in range(i):
          value -= fac_sh[base + k * n + i] * fac_sh[base + k * n + j]
        _st_fac_sh(fac_sh, base + i * n + j, value * diagonal_inv)
    _sync()
  for r in range(nmax):
    if active and r < n and j == n - 1 - r:
      value = fac_sh[xbase + j]
      for k in range(j + 1, n):
        value -= fac_sh[base + j * n + k] * fac_sh[xbase + k]
      _st_fac_sh(fac_sh, xbase + j, value / fac_sh[base + j * n + j])
    _sync()


@wp.func
def _dense_factor_solve_warp(fac_sh: wp.tile[float, _FAC_SIZE], active: bool, base: int, n: int, j: int, nmax: int):
  """``_dense_factor_solve`` for blocks whose dof columns all lie within one warp.

  Same arithmetic and order (bitwise results); the column steps are separated by ``__syncwarp``
  instead of block barriers, so every thread of every warp must still call it with the same
  ``nmax`` while only the warps holding active columns do work. Callers guarantee that no active
  block straddles a warp boundary.
  """
  xbase = base + n * n
  for i in range(nmax):
    if active and i < n and j >= i:
      diagonal_value = fac_sh[base + i * n + i]
      for k in range(i):
        factor = fac_sh[base + k * n + i]
        diagonal_value -= factor * factor
      diagonal_factor = wp.sqrt(diagonal_value)
      diagonal_inv = 1.0 / diagonal_factor
      if j == i:
        rhs_value = fac_sh[xbase + i]
        for k in range(i):
          rhs_value -= fac_sh[base + k * n + i] * fac_sh[xbase + k]
        _st_fac_sh(fac_sh, base + i * n + i, diagonal_factor)
        _st_fac_sh(fac_sh, xbase + i, rhs_value * diagonal_inv)
      else:
        value = fac_sh[base + j * n + i]
        for k in range(i):
          value -= fac_sh[base + k * n + i] * fac_sh[base + k * n + j]
        _st_fac_sh(fac_sh, base + i * n + j, value * diagonal_inv)
    _syncwarp()
  for r in range(nmax):
    if active and r < n and j == n - 1 - r:
      value = fac_sh[xbase + j]
      for k in range(j + 1, n):
        value -= fac_sh[base + j * n + k] * fac_sh[xbase + k]
      _st_fac_sh(fac_sh, xbase + j, value / fac_sh[base + j * n + j])
    _syncwarp()


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
  b_tree: int,
  b_mocap_root: int,
  d_body: int,
  d_tree: int,
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

  ``b_tree``/``b_mocap_root`` are ``body_treeid[tid]`` and ``body_mocapid[body_rootid[tid]]`` of the
  calling thread's body, ``d_body``/``d_tree`` the body and tree of its dof (caller-prefetched).
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
    state = int(SleepState.STATIC)
    if b_tree < 0:
      if b_mocap_root >= 0:
        state = int(SleepState.AWAKE)
    elif tree_asleep_sh[b_tree] < 0:
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
    if d_tree >= 0 and body_awake_sh[d_body] == SleepState.AWAKE:
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

  The result freezes facts that live in mutable device arrays of the built model (``jnt_type``,
  ``body_parentid``, ``eq_type``, the actuator transmission/dynamics/gain/bias types and the tree
  layout): mutating those after ``put_model`` is unsupported on the fused path.

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
  # the fused constraint kernel adds no surface velocities (constraint._add_surface_vel reads the
  # geom frames, which the publish opt-in may leave stale)
  if m.flg_surfacevel:
    return False

  jnt_type = np.asarray(mjm.jnt_type)
  if not np.isin(jnt_type, (JointType.FREE, JointType.HINGE, JointType.SLIDE)).all():
    return False
  if mjm.nbody > 1 and np.asarray(mjm.body_jntnum).max() > 1:
    return False
  if mjm.neq and not (np.asarray(mjm.eq_type) == EqType.JOINT).all():
    return False
  # the fused constraint kernel allocates the equality rows with one block-wide scan and stashes
  # the equality wake data in its per-body arena
  if mjm.neq > NBODY_CAP:
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
  if tree_dofnum.min() <= 0 or tree_dofnum.max() > NVTREE_CAP or (tree_dofnum > NVTREE_SMALL).sum() > NBIG_CAP:
    return False
  # every dof belongs to a tree: the kernels index tree state with dof_treeid unguarded
  if (np.asarray(mjm.body_treeid)[np.asarray(mjm.dof_bodyid)] < 0).any():
    return False
  if (np.asarray(m.qLD_block_adr) == Q_LD_BLOCK_SPARSE).any():
    return False

  return _dfs_contiguous(np.asarray(mjm.body_parentid))


def static_tables(mjm, m: Model) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Packed topology tables of the fused forward kernel, built once from host data in put_model.

  The kernel loads one 16-byte word per table row instead of a dependent chain of scalar lookups
  (body -> joint -> qpos address). Rows are int32 vec4:

  * ``body_info[b, 0] = (parentid, treeid, rootid, mocapid)``
  * ``body_info[b, 1] = (jntadr, jntnum, dofadr, dofnum)``
  * ``body_info[b, 2] = (type of the body's joint or -1, its qpos address, dynamic, 0)`` where
    ``dynamic`` marks bodies whose geoms move (not welded to the world, or under a mocap root)
  * ``dof_info[d, 0] = (bodyid, treeid, jntid, jnt_type | actgravcomp << 8 | actfrclimited << 9)``
  * ``dof_info[d, 1] = (jnt_qposadr, jnt_dofadr, M_rownnz, M_rowadr)``
  * ``act_info[u] = (dofadr and qpos address of the transmission joint,
    gaintype | biastype << 4 | ctrllimited << 8 | forcelimited << 9, 0)``

  The tables freeze the same topology as :func:`static_eligible` and are only meaningful for
  models that predicate accepts (multi-joint bodies keep their first joint, non-joint
  transmissions their first target clipped to the joint range).

  Args:
    mjm: The MuJoCo model (host arrays).
    m: The partially built MJWarp model; ``M_rownnz``/``M_rowadr`` are read as host arrays.

  Returns:
    ``(body_info, dof_info, act_info)`` with shapes ``(nbody, 3, 4)``, ``(nv, 2, 4)``, ``(nu, 4)``.
  """
  nbody, nv, nu, njnt = mjm.nbody, mjm.nv, mjm.nu, mjm.njnt
  i32 = lambda a: np.asarray(a, dtype=np.int32)  # noqa: E731
  parentid, treeid, rootid = i32(mjm.body_parentid), i32(mjm.body_treeid), i32(mjm.body_rootid)
  mocapid, weldid = i32(mjm.body_mocapid), i32(mjm.body_weldid)
  jntadr, jntnum, dofadr, dofnum = i32(mjm.body_jntadr), i32(mjm.body_jntnum), i32(mjm.body_dofadr), i32(mjm.body_dofnum)
  jnt_type, jnt_qposadr, jnt_dofadr = i32(mjm.jnt_type), i32(mjm.jnt_qposadr), i32(mjm.jnt_dofadr)
  zeros_b = np.zeros(nbody, dtype=np.int32)

  body_info = np.zeros((nbody, 3, 4), dtype=np.int32)
  body_info[:, 0] = np.stack([parentid, treeid, rootid, mocapid], axis=1)
  body_info[:, 1] = np.stack([jntadr, jntnum, dofadr, dofnum], axis=1)
  has_jnt = jntnum > 0
  jnt_of_body = np.where(has_jnt, jntadr, 0)
  j_type = np.where(has_jnt, jnt_type[jnt_of_body], -1).astype(np.int32) if njnt else zeros_b - 1
  j_qadr = np.where(has_jnt, jnt_qposadr[jnt_of_body], 0).astype(np.int32) if njnt else zeros_b
  dynamic = ((weldid != 0) | (mocapid[rootid] != -1)).astype(np.int32)
  body_info[:, 2] = np.stack([j_type, j_qadr, dynamic, zeros_b], axis=1)

  dof_info = np.zeros((nv, 2, 4), dtype=np.int32)
  if nv:
    dof_body, dof_jnt = i32(mjm.dof_bodyid), i32(mjm.dof_jntid)
    flags = jnt_type[dof_jnt] | (i32(mjm.jnt_actgravcomp)[dof_jnt] << 8) | (i32(mjm.jnt_actfrclimited)[dof_jnt] << 9)
    dof_info[:, 0] = np.stack([dof_body, treeid[dof_body], dof_jnt, flags], axis=1)
    dof_info[:, 1] = np.stack([jnt_qposadr[dof_jnt], jnt_dofadr[dof_jnt], i32(m.M_rownnz), i32(m.M_rowadr)], axis=1)

  act_info = np.zeros((nu, 4), dtype=np.int32)
  if nu and njnt:
    jnt = np.clip(i32(mjm.actuator_trnid)[:, 0], 0, njnt - 1)
    flags = (
      i32(mjm.actuator_gaintype)
      | (i32(mjm.actuator_biastype) << 4)
      | (i32(mjm.actuator_ctrllimited) << 8)
      | (i32(mjm.actuator_forcelimited) << 9)
    )
    act_info[:] = np.stack([jnt_dofadr[jnt], jnt_qposadr[jnt], flags, np.zeros(nu, dtype=np.int32)], axis=1)
  return body_info, dof_info, act_info


def fused_world(m: Model, d: Data) -> bool:
  """Return whether the fused per-world forward kernels apply to ``(m, d)``.

  The structural part comes from ``m.fused_world_static`` (host data in put_model); option flags
  (including the ``Option.fused_world`` host switch) and runtime toggles (such as
  ``sensor_rne_postconstraint``, which Newton flips at runtime) are re-evaluated on every call. Only
  host state is read, so the predicate is safe under graph capture.
  """
  if not getattr(m, "fused_world_static", False):
    return False
  opt = m.opt
  if not getattr(opt, "fused_world", True):
    return False
  if opt.solver != SolverType.NEWTON or opt.cone != ConeType.PYRAMIDAL or opt.integrator != IntegratorType.IMPLICITFAST:
    return False
  if opt.run_collision_detection:
    return False
  if not (opt.enableflags & EnableBit.SLEEP) or (opt.disableflags & DisableBit.ISLAND):
    return False
  if m.sensor_rne_postconstraint:
    return False
  # post_position only reads the poses forward_a publishes; the other callbacks may read fused state
  if m.callback.observes_derived_state():
    return False
  if d.nworld == 0 or not d.qpos.device.is_cuda:
    return False
  if d.qLD.shape[1] != m.qLD_block_total:
    return False
  return True


# Data fields that only forward_a derives and no later stage of a fused step reads: forward_m/b/c,
# the constraint solver, camlight, the energy terms, the post_position callback and forward_worlds
# consume the state (qpos, qvel, sleep arrays), xpos/xquat/xipos, subtree_com, cinert, cdof, M,
# cvel, the qfrc_* forces and actuator_force, which are published unconditionally. When
# publish_derived(m) is False, forward_a and forward_worlds leave these fields stale; the
# finalizing forward_c rebuilds DERIVED_FIELDS_REFRESHED on the post-sleep velocities, so after a
# finalizing step only the frames, crb and the actuator length/moment stay stale.
DERIVED_FIELDS = (
  "xmat",
  "ximat",
  "xanchor",
  "xaxis",
  "geom_xpos",
  "geom_xmat",
  "site_xpos",
  "site_xmat",
  "crb",
  "actuator_length",
  "moment_rownnz",
  "moment_rowadr",
  "moment_colind",
  "actuator_moment",
  "actuator_velocity",
  "cdof_dot",
  "qfrc_spring",
  "qfrc_damper",
  "qfrc_adhesion",
  "cacc",
  "cfrc_int",
  "cvel",
  "qLD",
)
# ``efc`` row fields that only sensors read (constraint._efc_row publishes them for sensor_acc)
DERIVED_EFC_FIELDS = ("pos", "margin", "vel", "Jqvel")
# rebuilt by the finalizing forward_c refresh when the derived fields are published; otherwise the
# refresh is skipped entirely and these (with qfrc_passive/qfrc_bias) keep their pre-sleep values
DERIVED_FIELDS_REFRESHED = ("actuator_velocity", "cdof_dot", "qfrc_spring", "qfrc_damper", "cacc", "cfrc_int")


def publish_derived(m: Model) -> bool:
  """Return whether the fused forward publishes :data:`DERIVED_FIELDS`.

  False only when ``Option.fused_world_publish_derived`` is False and the model has no sensors
  (sensors read the derived frames, velocities and accelerations). Only host state is read, so the
  helper is safe under graph capture. Callers that read a field of :data:`DERIVED_FIELDS` or
  :data:`DERIVED_EFC_FIELDS` from ``Data`` after a step (Newton's ``body_qdd``/``body_parent_f``
  conversion reads ``cacc`` and ``cfrc_int``) must leave the option on. When False the finalizing
  post-sleep refresh is skipped as well, so ``qfrc_passive`` and ``qfrc_bias`` keep the pre-sleep
  values of the last substep (the next forward recomputes them).
  """
  return bool(getattr(m.opt, "fused_world_publish_derived", True) or m.nsensor > 0)


# resident CTAs per SM requested from the compiler for the fused position/velocity kernel: 7 CTAs/SM
# hold 1024 worlds in one wave on 170 SMs (<= 72 registers; the shared arenas stay under the 13 KB
# that seven CTAs can share on a 100 KB SM)
_FORWARD_A_MIN_BLOCKS = 7


def _pow2_at_least(n: int) -> int:
  """Smallest power of two >= max(n, 1) (tile shapes must be positive)."""
  return 1 << max(0, (n - 1).bit_length())


@cache_kernel
def _forward_a_kernel(NB: int, NV: int, NU: int, NT: int, PUBLISH: bool, SUBSET: bool):
  """forward_a kernel factory.

  ``PUBLISH`` compiles the derived publishes in (see ``Option.fused_world_publish_derived``), so the
  default variant carries no runtime gates. ``SUBSET`` makes CTA ``slot`` serve world
  ``world_ids[slot]`` (optionally bounded by a device count) instead of world ``slot``; the branch
  is resolved at code generation, so the full-grid kernel is unchanged.
  """
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
  def _st_body_f(values: wp.tile[float, NB], index: int, value: float): ...

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

  @wp.kernel(module="unique", enable_backward=False, grid_stride=False, launch_bounds=(BLOCK, _FORWARD_A_MIN_BLOCKS))
  def kernel(
    # Model:
    nbody: int,
    nv: int,
    nu: int,
    ntree: int,
    ngeom: int,
    nsite: int,
    has_gravcomp: int,
    run_wake: int,
    opt_disableflags: int,
    opt_gravity: wp.array[wp.vec3],
    qpos0: wp.array2d[float],
    qpos_spring: wp.array2d[float],
    body_info: wp.array2d[wp.vec4i],
    dof_info: wp.array2d[wp.vec4i],
    act_info: wp.array[wp.vec4i],
    body_pos: wp.array2d[wp.vec3],
    body_quat: wp.array2d[wp.quat],
    body_ipos: wp.array2d[wp.vec3],
    body_iquat: wp.array2d[wp.quat],
    body_mass: wp.array2d[float],
    body_subtreemass: wp.array2d[float],
    body_inertia: wp.array2d[wp.vec3],
    body_gravcomp: wp.array2d[float],
    jnt_pos: wp.array2d[wp.vec3],
    jnt_axis: wp.array2d[wp.vec3],
    jnt_stiffness: wp.array2d[float],
    jnt_stiffnesspoly: wp.array2d[wp.vec2],
    jnt_actfrcrange: wp.array2d[wp.vec2],
    dof_armature: wp.array2d[float],
    dof_damping: wp.array2d[float],
    dof_dampingpoly: wp.array2d[wp.vec2],
    tree_sleep_policy: wp.array[int],
    geom_bodyid: wp.array[int],
    geom_pos: wp.array2d[wp.vec3],
    geom_quat: wp.array2d[wp.quat],
    site_bodyid: wp.array[int],
    site_pos: wp.array2d[wp.vec3],
    site_quat: wp.array2d[wp.quat],
    actuator_gear: wp.array2d[wp.spatial_vector],
    actuator_gainprm: wp.array2d[vec10],
    actuator_biasprm: wp.array2d[vec10],
    actuator_ctrlrange: wp.array2d[wp.vec2],
    actuator_forcerange: wp.array2d[wp.vec2],
    # Data in:
    qpos_in: wp.array2d[float],
    qvel_in: wp.array2d[float],
    qfrc_applied_in: wp.array2d[float],
    xfrc_applied_in: wp.array2d[wp.spatial_vector],
    mocap_pos_in: wp.array2d[wp.vec3],
    mocap_quat_in: wp.array2d[wp.quat],
    ctrl_in: wp.array2d[float],
    # In:
    world_ids: wp.array[int],
    world_count: wp.array[int],
    # Data out:
    tree_asleep_out: wp.array2d[int],
    tree_awake_out: wp.array2d[int],
    world_con_count_out: wp.array[int],
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
    if wp.static(SUBSET):
      # uniform per CTA: no thread of an exiting CTA reaches a barrier below
      if world_count:
        if worldid >= world_count[0]:
          return
      worldid = world_ids[worldid]

    # shared arenas (per body / per dof / per actuator / per tree); several are reused by a later
    # phase once their first contents are dead, see the phase comments
    xpos_sh = wp.tile_empty(shape=(NB,), dtype=wp.vec3, storage="shared")  # P2-P2b: xpos; P3 on: subtree_com
    xquat_sh = wp.tile_empty(shape=(NB,), dtype=wp.quat, storage="shared")
    mcom_sh = wp.tile_empty(shape=(NB,), dtype=wp.vec3, storage="shared")  # P2b-P3: mass * xipos; P3b on: xipos - com
    cinert_sh = wp.tile_empty(shape=(NB,), dtype=vec10, storage="shared")
    cvel_sh = wp.tile_empty(shape=(NB,), dtype=wp.spatial_vector, storage="shared")  # P8: cfrc_int
    cacc_sh = wp.tile_empty(shape=(NB,), dtype=wp.spatial_vector, storage="shared")  # P8: body-local force
    # per-body packed layout: parent | subtree_end << 8 | dynamic << 16 | (dofadr + 1) << 17 |
    # dofnum << 25
    binfo_sh = wp.tile_empty(shape=(NB,), dtype=int, storage="shared")
    mass_sh = wp.tile_empty(shape=(NB,), dtype=float, storage="shared")
    gcomp_sh = wp.tile_empty(shape=(NB,), dtype=float, storage="shared")
    cdof_sh = wp.tile_empty(shape=(NV,), dtype=wp.spatial_vector, storage="shared")
    qvel_sh = wp.tile_empty(shape=(NV,), dtype=float, storage="shared")
    act_force_sh = wp.tile_empty(shape=(NU,), dtype=float, storage="shared")
    act_dof_sh = wp.tile_empty(shape=(NU,), dtype=int, storage="shared")
    tree_asleep_sh = wp.tile_empty(shape=(NT,), dtype=int, storage="shared")
    misc_sh = wp.tile_empty(shape=(16,), dtype=int, storage="shared")

    lane = tid & 31
    warp = tid >> 5

    is_body = tid < nbody
    is_dof = tid < nv
    is_act = tid < nu  # nu <= NU <= BLOCK: one actuator per thread
    is_tree = tid < ntree

    gravity_enabled = (opt_disableflags & DisableBit.GRAVITY) == 0
    dsbl_spring = (opt_disableflags & DisableBit.SPRING) != 0
    dsbl_damper = (opt_disableflags & DisableBit.DAMPER) != 0
    gravity = opt_gravity[worldid % opt_gravity.shape[0]]
    if tid == 0:
      # the contact bucket pass between forward_a and forward_m appends to this count
      world_con_count_out[worldid] = 0
      # roots of the com_vel / cacc traversal (P6)
      _st_body_sv(cvel_sh, 0, wp.spatial_vector())
      cacc0 = wp.spatial_vector()
      if gravity_enabled:
        cacc0 = wp.spatial_vector(wp.vec3(0.0), -gravity)
      _st_body_sv(cacc_sh, 0, cacc0)

    # -------------------------------------------------------- P0: prefetch (two dependent rounds)
    # packed topology tables (static_tables) first, then the per-world parameters they address.
    # Bodies have at most one joint (fused_world()); the body frame holds the free joint pose for
    # free bodies and the mocap pose for mocap bodies
    b_parent = int(0)
    b_tree = int(-1)
    b_root = int(0)
    b_jntadr = int(-1)
    b_jntnum = int(0)
    b_dofadr = int(-1)
    b_dofnum = int(0)
    b_mass = float(0.0)
    b_xfrc_nz = int(0)
    b_dyn = int(0)
    j_type = int(-1)
    j_axis = wp.vec3(0.0)
    j_pos = wp.vec3(0.0)
    j_q = float(0.0)
    j_q0 = float(0.0)
    b_pos = wp.vec3(0.0)
    b_quat = wp.quat(1.0, 0.0, 0.0, 0.0)
    if is_body:
      binfo0 = body_info[tid, 0]
      binfo1 = body_info[tid, 1]
      binfo2 = body_info[tid, 2]
      b_parent = binfo0[0]
      b_tree = binfo0[1]
      b_root = binfo0[2]
      b_mocap = binfo0[3]
      b_jntadr = binfo1[0]
      b_jntnum = binfo1[1]
      b_dofadr = binfo1[2]
      b_dofnum = binfo1[3]
      j_type = binfo2[0]
      j_qadr = binfo2[1]
      # geoms of world-attached bodies are static unless the body descends from a mocap root
      b_dyn = binfo2[2]
      b_mass = body_mass[worldid % body_mass.shape[0], tid]
      b_pos = body_pos[worldid % body_pos.shape[0], tid]
      b_quat = body_quat[worldid % body_quat.shape[0], tid]
      _st_body_f(mass_sh, tid, b_mass)
      _st_body_f(gcomp_sh, tid, body_gravcomp[worldid % body_gravcomp.shape[0], tid])
      _st_body_i(binfo_sh, tid, b_parent)
      if run_wake != 0:
        if not (xfrc_applied_in[worldid, tid] == wp.spatial_vector()):
          b_xfrc_nz = 1
      if b_jntnum == 1:
        j_axis = jnt_axis[worldid % jnt_axis.shape[0], b_jntadr]
        j_pos = jnt_pos[worldid % jnt_pos.shape[0], b_jntadr]
        if j_type == JointType.FREE:
          qpos = qpos_in[worldid]
          b_pos = wp.vec3(qpos[j_qadr], qpos[j_qadr + 1], qpos[j_qadr + 2])
          b_quat = wp.quat(qpos[j_qadr + 3], qpos[j_qadr + 4], qpos[j_qadr + 5], qpos[j_qadr + 6])
        else:
          j_q = qpos_in[worldid, j_qadr]
          j_q0 = qpos0[worldid % qpos0.shape[0], j_qadr]
      elif b_mocap >= 0:
        b_pos = mocap_pos_in[worldid, b_mocap]
        b_quat = mocap_quat_in[worldid, b_mocap]
      if tid == 0:
        # the world body keeps whatever pose Data carries (kinematics never writes body 0)
        _st_body_v3(xpos_sh, 0, xpos_out[worldid, 0])
        _st_body_q(xquat_sh, 0, xquat_out[worldid, 0])
    # dof-owned parameters; the dof's joint is its body's single joint
    d_body = int(0)
    d_tree = int(-1)
    d_jnt = int(0)
    d_qvel = float(0.0)
    d_qfrc_nz = int(0)
    dj_type = int(0)
    dj_qadr = int(0)
    dj_dofadr = int(0)
    d_rownnz = int(0)
    d_rowadr = int(0)
    actgravcomp = int(0)
    d_frclimited = int(0)
    if is_dof:
      dinfo0 = dof_info[tid, 0]
      dinfo1 = dof_info[tid, 1]
      d_body = dinfo0[0]
      d_tree = dinfo0[1]
      d_jnt = dinfo0[2]
      dj_type = dinfo0[3] & 0xFF
      actgravcomp = (dinfo0[3] >> 8) & 1
      d_frclimited = (dinfo0[3] >> 9) & 1
      dj_qadr = dinfo1[0]
      dj_dofadr = dinfo1[1]
      d_rownnz = dinfo1[2]
      d_rowadr = dinfo1[3]
      d_qvel = qvel_in[worldid, tid]
      if run_wake != 0:
        if qfrc_applied_in[worldid, tid] != 0.0:
          d_qfrc_nz = 1
      _st_dof_f(qvel_sh, tid, d_qvel)
    # actuator-owned parameters (joint transmissions on hinge/slide joints)
    a_vadr = int(0)
    a_gear = float(0.0)
    a_length = float(0.0)
    a_flags = int(0)
    if is_act:
      ainfo = act_info[tid]
      a_vadr = ainfo[0]
      a_flags = ainfo[2]
      a_gear = actuator_gear[worldid % actuator_gear.shape[0], tid][0]
      a_length = qpos_in[worldid, ainfo[1]] * a_gear
      _st_act_i(act_dof_sh, tid, a_vadr)
    # tree-owned state
    t_awake = int(0)
    if is_tree:
      _st_tree_i(tree_asleep_sh, tid, tree_asleep_out[worldid, tid])
      t_awake = tree_awake_out[worldid, tid]
    # first geom / site of this thread (the P2b loops load the rest)
    g_body = int(0)
    if tid < ngeom:
      g_body = geom_bodyid[tid]
    s_body = int(0)
    if tid < nsite:
      s_body = site_bodyid[tid]

    # per-tree wake masks (ntree <= 32): applied Cartesian force, applied generalized force,
    # velocity
    if run_wake != 0:
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

    # depth-first numbering: the subtree of tid is [tid, subtree_end). The parent field is read
    # masked, so the packed rewrite of this thread's own slot below is safe while other threads
    # still scan parents.
    subtree_end = int(0)
    if is_body:
      subtree_end = nbody
      if tid > 0:
        subtree_end = tid + 1
        while subtree_end < nbody:
          if (binfo_sh[subtree_end] & 0xFF) < tid:
            break
          subtree_end += 1
      _st_body_i(binfo_sh, tid, b_parent | (subtree_end << 8) | (b_dyn << 16) | ((b_dofadr + 1) << 17) | (b_dofnum << 25))

    # ---------------------------------------------------------------- P1: sleep.wake
    # skipped when the caller declares it already ran sleep.wake on these inputs
    if run_wake != 0 and is_tree:
      if tree_asleep_sh[tid] >= 0:
        wake_bits = wp.uint32(0)
        for w in range(4):
          wake_bits |= wp.uint32(misc_sh[w]) | wp.uint32(misc_sh[4 + w]) | wp.uint32(misc_sh[8 + w])
        wake = int(0)
        if t_awake == 1:
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

    # ----------------------------------------------------------- P2: kinematics (ancestor walk)
    # parameters of P2b-P4 are loaded first so that the barriers hide their latency
    b_ipos = wp.vec3(0.0)
    b_iquat = wp.quat(1.0, 0.0, 0.0, 0.0)
    b_inertia = wp.vec3(0.0)
    b_submass = float(0.0)
    if is_body:
      b_ipos = body_ipos[worldid % body_ipos.shape[0], tid]
      b_iquat = body_iquat[worldid % body_iquat.shape[0], tid]
      b_inertia = body_inertia[worldid % body_inertia.shape[0], tid]
      b_submass = body_subtreemass[worldid % body_subtreemass.shape[0], tid]
    d_armature = float(0.0)
    if is_dof:
      d_armature = dof_armature[worldid % dof_armature.shape[0], tid]
    # Local transform of each body relative to its parent, with the joint folded in: hinge
    # xquat = xquat_p * b_quat * j_quat and xpos = xpos_p + R_p (b_pos + R_b (j_pos - R_j j_pos)),
    # slide xpos = xpos_p + R_p (b_pos + R_b j_axis * dq). Free bodies carry their world pose.
    j_dq = j_q - j_q0
    j_quat = wp.quat(1.0, 0.0, 0.0, 0.0)
    if is_body and j_type == JointType.HINGE:
      j_quat = math.axis_angle_to_quat(j_axis, j_dq)
    is_free = is_body and b_jntnum == 1 and j_type == JointType.FREE
    xpos = b_pos
    xquat = b_quat
    if is_body and tid > 0:
      if is_free:
        xquat = wp.normalize(b_quat)
      elif b_jntnum == 1:
        if j_type == JointType.SLIDE:
          xpos = b_pos + math.rot_vec_quat(j_axis, b_quat) * j_dq
        elif j_type == JointType.HINGE:
          xquat = math.mul_quat(b_quat, j_quat)
          xpos = b_pos + math.rot_vec_quat(j_pos - math.rot_vec_quat(j_pos, j_quat), b_quat)
      _st_body_v3(xpos_sh, tid, xpos)
      _st_body_q(xquat_sh, tid, xquat)
    _sync()
    # P1b: update_sleep (trees). tree_awake feeds wake_equality in forward_m, which then
    # publishes the body/dof awake arrays
    if is_tree:
      asleep = tree_asleep_sh[tid]
      tree_asleep_out[worldid, tid] = asleep
      awake_flag = int(0)
      if asleep < 0:
        awake_flag = 1
      tree_awake_out[worldid, tid] = awake_flag
    # Each body composes the local transforms along its ancestor chain up to the world pose: one
    # dependent shared read per level instead of one block barrier per level. The composition
    # order (leaf to root, normalized once) differs from the stock level recursion by fp32
    # rounding only.
    if is_body and tid > 0 and not is_free:
      p = int(b_parent)
      while p != 0:
        pquat = xquat_sh[p]
        xpos = xpos_sh[p] + math.rot_vec_quat(xpos, pquat)
        xquat = math.mul_quat(pquat, xquat)
        p = binfo_sh[p] & 0xFF
      pquat = xquat_sh[0]
      xpos = xpos_sh[0] + math.rot_vec_quat(xpos, pquat)
      xquat = wp.normalize(math.mul_quat(pquat, xquat))
    # joint frame from the body's own pose: xpos = xanchor - R j_pos for hinges (whose rotation
    # leaves the axis invariant); the slide offset is removed from the anchor
    xanchor = wp.vec3(0.0)
    xaxis = wp.vec3(0.0)
    if is_body and b_jntnum == 1:
      if is_free:
        xanchor = xpos
        xaxis = j_axis
      else:
        xaxis = math.rot_vec_quat(j_axis, xquat)
        xanchor = math.rot_vec_quat(j_pos, xquat) + xpos
        if j_type == JointType.SLIDE:
          xanchor -= xaxis * j_dq
    _sync()
    if is_body and tid > 0:
      _st_body_v3(xpos_sh, tid, xpos)
      _st_body_q(xquat_sh, tid, xquat)
    _sync()

    # ------------------------------------------------------- P2b: body frames, joints, geoms, sites
    xipos = wp.vec3(0.0)
    if is_body:
      if tid == 0:
        xpos = xpos_sh[0]
        xquat = xquat_sh[0]
      xipos = xpos + math.rot_vec_quat(b_ipos, xquat)
      xpos_out[worldid, tid] = xpos
      _st_q(xquat_out, worldid, tid, xquat)
      xipos_out[worldid, tid] = xipos
      _st_body_v3(mcom_sh, tid, xipos * b_mass)
      if PUBLISH:
        _st_m33(xmat_out, worldid, tid, math.quat_to_mat(xquat))
        _st_m33(ximat_out, worldid, tid, math.quat_to_mat(math.mul_quat(xquat, b_iquat)))
        if b_jntnum == 1:
          xanchor_out[worldid, b_jntadr] = xanchor
          xaxis_out[worldid, b_jntadr] = xaxis
    # geom and site frames are derived publishes too
    ngeom_pub = int(0)
    nsite_pub = int(0)
    if PUBLISH:
      ngeom_pub = ngeom
      nsite_pub = nsite
    for geomid in range(tid, ngeom_pub, BLOCK):
      bodyid = g_body
      if geomid != tid:
        bodyid = geom_bodyid[geomid]
      if (binfo_sh[bodyid] >> 16) != 0:
        gpos = xpos_sh[bodyid]
        gquat = xquat_sh[bodyid]
        geom_xpos_out[worldid, geomid] = gpos + math.rot_vec_quat(geom_pos[worldid % geom_pos.shape[0], geomid], gquat)
        _st_m33(
          geom_xmat_out,
          worldid,
          geomid,
          math.quat_to_mat(math.mul_quat(gquat, geom_quat[worldid % geom_quat.shape[0], geomid])),
        )
    for siteid in range(tid, nsite_pub, BLOCK):
      bodyid = s_body
      if siteid != tid:
        bodyid = site_bodyid[siteid]
      gpos = xpos_sh[bodyid]
      gquat = xquat_sh[bodyid]
      site_xpos_out[worldid, siteid] = gpos + math.rot_vec_quat(site_pos[worldid % site_pos.shape[0], siteid], gquat)
      _st_m33(
        site_xmat_out, worldid, siteid, math.quat_to_mat(math.mul_quat(gquat, site_quat[worldid % site_quat.shape[0], siteid]))
      )
    _sync()

    # ---------------------------------------------------------------- P3: subtree_com
    # gathered from the mass-weighted positions; the result lives in the (now dead) xpos arena
    com = wp.vec3(0.0)
    if is_body:
      com = mcom_sh[tid]
      if tid > 0:
        for c in range(tid + 1, subtree_end):
          com += mcom_sh[c]
    # the world body's gather spans every body: the first warp shares it (lane-strided partial sums)
    if warp == 0:
      part = wp.vec3(0.0)
      for c in range(lane + 1, nbody, 32):
        part += mcom_sh[c]
      part = wp.vec3(_warp_sum(part[0]), _warp_sum(part[1]), _warp_sum(part[2]))
      if tid == 0:
        com += part
    if is_body:
      if b_submass != 0.0:
        com = com / b_submass
      _st_body_v3(xpos_sh, tid, com)
      subtree_com_out[worldid, tid] = com
    _sync()

    # ---------------------------------------------------------------- P3b: cinert, cdof
    if is_body:
      mat = math.quat_to_mat(math.mul_quat(xquat, b_iquat))
      dif = xipos - xpos_sh[b_root]
      # express inertia in com-based frame (mju_inertCom)
      res = vec10()
      tmp = mat @ wp.diag(b_inertia) @ wp.transpose(mat)
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
      _st_v10(cinert_out, worldid, tid, res)
      # lever arm of the gravity compensation force (P7) replaces the dead mass-weighted position
      _st_body_v3(mcom_sh, tid, dif)
      if b_jntnum == 1:
        xmat = wp.transpose(math.quat_to_mat(xquat))
        offset = xpos_sh[b_root] - xanchor
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
    if is_body and PUBLISH:
      crb = cinert_sh[tid]
      # the world body never accumulates its children
      if tid > 0:
        for c in range(tid + 1, subtree_end):
          crb += cinert_sh[c]
      _st_v10(crb_out, worldid, tid, crb)
    cdof = wp.spatial_vector()
    d_send = int(0)
    if is_dof:
      cdof = cdof_sh[tid]
      _st_sv(cdof_out, worldid, tid, cdof)
      # composite inertia of the dof's body, gathered in the same order as the body's crb above
      d_send = (binfo_sh[d_body] >> 8) & 0xFF
      crb_d = cinert_sh[d_body]
      for c in range(d_body + 1, d_send):
        crb_d += cinert_sh[c]
      madr_ij = d_rowadr + d_rownnz - 1
      buf = math.inert_vec(crb_d, cdof)
      # diagonal: armature inertia plus the composite inertia term
      M_out[worldid, madr_ij] = d_armature + wp.dot(cdof, buf)
      # ancestors fill the rest of the CSR row, walking dof_parentid through the packed body
      # layout: the previous dof of the same joint, then the last dof of the closest ancestor body
      # with dofs. "Simple" dofs (rownnz == 1) store only the diagonal, so the walk is bounded by
      # the row length
      dofid = int(tid)
      info = binfo_sh[d_body]
      jstart = ((info >> 17) & 0xFF) - 1
      for _k in range(d_rownnz - 1):
        madr_ij -= 1
        if dofid > jstart:
          dofid -= 1
        else:
          body = info & 0xFF
          info = binfo_sh[body]
          while body != 0 and ((info >> 25) & 0x7) == 0:
            body = info & 0xFF
            info = binfo_sh[body]
          jstart = ((info >> 17) & 0xFF) - 1
          dofid = jstart + ((info >> 25) & 0x7) - 1
        M_out[worldid, madr_ij] = wp.dot(cdof_sh[dofid], buf)

    # ---------------------------------------------------------- P5: transmission, actuator velocity
    a_velocity = float(0.0)
    if is_act:
      a_velocity = a_gear * qvel_sh[a_vadr]
      if PUBLISH:
        actuator_length_out[worldid, tid] = a_length
        moment_rownnz_out[worldid, tid] = 1
        moment_rowadr_out[worldid, tid] = tid
        moment_colind_out[worldid, tid] = a_vadr
        actuator_moment_out[worldid, tid] = a_gear
        actuator_velocity_out[worldid, tid] = a_velocity

    # ----------------------------------------------------- P6: com_vel and cacc (ancestor sums)
    # the passive-force parameters of P7 are loaded first; the barriers hide their latency
    stiffness = float(0.0)
    spoly = wp.vec2(0.0)
    damping = float(0.0)
    dpoly = wp.vec2(0.0)
    q = float(0.0)
    q_spring = float(0.0)
    if is_dof:
      if not (dsbl_spring and dsbl_damper):
        stiffness = jnt_stiffness[worldid % jnt_stiffness.shape[0], d_jnt]
        spoly = jnt_stiffnesspoly[worldid % jnt_stiffnesspoly.shape[0], d_jnt]
        # the joint's first dof supplies the damping of all its dofs (as _spring_damper_dof_passive)
        damping = dof_damping[worldid % dof_damping.shape[0], dj_dofadr]
        dpoly = dof_dampingpoly[worldid % dof_dampingpoly.shape[0], dj_dofadr]
        if dj_type != JointType.FREE:
          q = qpos_in[worldid, dj_qadr]
          q_spring = qpos_spring[worldid % qpos_spring.shape[0], dj_qadr]
    # Each body's own velocity term v = sum_k cdof_k qvel_k; free bodies (children of the world,
    # whose cvel is zero) also finish their cdof_dot and acceleration term here.
    v_own = wp.spatial_vector()
    a_own = wp.spatial_vector()
    if is_body and tid > 0 and b_jntnum == 1:
      if j_type == JointType.FREE:
        for k in range(3):
          v_own += cdof_sh[b_dofadr + k] * qvel_sh[b_dofadr + k]
          if PUBLISH:
            _st_sv(cdof_dot_out, worldid, b_dofadr + k, wp.spatial_vector())
        for k in range(3, 6):
          cdof_dot = math.motion_cross(v_own, cdof_sh[b_dofadr + k])
          if PUBLISH:
            _st_sv(cdof_dot_out, worldid, b_dofadr + k, cdof_dot)
          a_own += cdof_dot * qvel_sh[b_dofadr + k]
        for k in range(3, 6):
          v_own += cdof_sh[b_dofadr + k] * qvel_sh[b_dofadr + k]
      else:
        v_own = cdof_sh[b_dofadr] * qvel_sh[b_dofadr]
    if is_body and tid > 0:
      _st_body_sv(cvel_sh, tid, v_own)
    _sync()
    # cvel_parent = sum of the ancestors' terms (cvel_sh[0] is zero), read along the chain instead
    # of one block barrier per level; the summation order (leaf to root) differs from the stock
    # level recursion by fp32 rounding only. cdof_dot of a hinge/slide dof uses the parent velocity.
    cvel = wp.spatial_vector()
    if is_body and tid > 0:
      p = int(b_parent)
      while p != 0:
        cvel += cvel_sh[p]
        p = binfo_sh[p] & 0xFF
      if b_jntnum == 1 and j_type != JointType.FREE:
        cdof_dot = math.motion_cross(cvel, cdof_sh[b_dofadr])
        if PUBLISH:
          _st_sv(cdof_dot_out, worldid, b_dofadr, cdof_dot)
        a_own = cdof_dot * qvel_sh[b_dofadr]
      cvel += v_own
    _sync()
    if is_body and tid > 0:
      _st_body_sv(cvel_sh, tid, cvel)
      _st_body_sv(cacc_sh, tid, a_own)
    _sync()
    # cacc = gravity root + the ancestors' terms + own term, same chain read
    cacc = wp.spatial_vector()
    if is_body:
      cacc = cacc_sh[0]
      if tid > 0:
        p = int(b_parent)
        while p != 0:
          cacc += cacc_sh[p]
          p = binfo_sh[p] & 0xFF
        cacc += a_own
    _sync()
    if is_body:
      if tid > 0:
        _st_body_sv(cacc_sh, tid, cacc)
      if PUBLISH:
        _st_sv(cvel_out, worldid, tid, cvel)
        _st_sv(cacc_out, worldid, tid, cacc)
    _sync()

    # ---------------------------------------------------------------- P7: passive (per dof)
    qfrc_spring = float(0.0)
    qfrc_damper = float(0.0)
    qfrc_gravcomp = float(0.0)
    qfrc_passive = float(0.0)
    if is_dof:
      if not (dsbl_spring and dsbl_damper):
        has_stiffness = (stiffness != 0.0 or spoly[0] != 0.0 or spoly[1] != 0.0) and not dsbl_spring
        has_damping = (damping != 0.0 or dpoly[0] != 0.0 or dpoly[1] != 0.0) and not dsbl_damper
        if dj_type == JointType.FREE:
          if has_stiffness:
            qpos_spring_id = worldid % qpos_spring.shape[0]
            k = tid - dj_dofadr
            if k < 3:
              dif = wp.vec3(
                qpos_in[worldid, dj_qadr + 0] - qpos_spring[qpos_spring_id, dj_qadr + 0],
                qpos_in[worldid, dj_qadr + 1] - qpos_spring[qpos_spring_id, dj_qadr + 1],
                qpos_in[worldid, dj_qadr + 2] - qpos_spring[qpos_spring_id, dj_qadr + 2],
              )
              kf = util_misc._poly_force(stiffness, spoly, wp.length(dif), 0)
              qfrc_spring = -kf * dif[k]
            else:
              rot = wp.quat(
                qpos_in[worldid, dj_qadr + 3],
                qpos_in[worldid, dj_qadr + 4],
                qpos_in[worldid, dj_qadr + 5],
                qpos_in[worldid, dj_qadr + 6],
              )
              rot = wp.normalize(rot)
              ref = wp.quat(
                qpos_spring[qpos_spring_id, dj_qadr + 3],
                qpos_spring[qpos_spring_id, dj_qadr + 4],
                qpos_spring[qpos_spring_id, dj_qadr + 5],
                qpos_spring[qpos_spring_id, dj_qadr + 6],
              )
              dif = math.quat_sub(rot, ref)
              k_rot = util_misc._poly_force(stiffness, spoly, wp.length(dif), 0)
              qfrc_spring = -k_rot * dif[k - 3]
          if has_damping:
            qfrc_damper = -d_qvel * util_misc._poly_force(damping, dpoly, d_qvel, 1)
        else:  # SLIDE, HINGE
          if has_stiffness:
            fdif = q - q_spring
            qfrc_spring = -fdif * util_misc._poly_force(stiffness, spoly, fdif, 0)
          if has_damping:
            qfrc_damper = -d_qvel * util_misc._poly_force(damping, dpoly, d_qvel, 1)

        # gravity compensation: compensated bodies in this dof's subtree (support.jac_dof via
        # body_isdofancestor), in body order
        if gravity_enabled and has_gravcomp != 0:
          cdof_ang = wp.spatial_top(cdof)
          cdof_lin = wp.spatial_bottom(cdof)
          for bodyid in range(d_body, d_send):
            gcomp = gcomp_sh[bodyid]
            if gcomp != 0.0:
              jacp = cdof_lin + wp.cross(cdof_ang, mcom_sh[bodyid])
              qfrc_gravcomp += wp.dot(jacp, -gravity * mass_sh[bodyid] * gcomp)

        qfrc_passive = qfrc_spring + qfrc_damper
        if gravity_enabled:
          if actgravcomp == 0:
            qfrc_passive += qfrc_gravcomp

      qfrc_gravcomp_out[worldid, tid] = qfrc_gravcomp
      qfrc_passive_out[worldid, tid] = qfrc_passive
      if PUBLISH:
        qfrc_spring_out[worldid, tid] = qfrc_spring
        qfrc_damper_out[worldid, tid] = qfrc_damper
        qfrc_adhesion_out[worldid, tid] = 0.0

    # ---------------------------------------------------------------- P8: rne (cfrc_int, qfrc_bias)
    # the actuation parameters of P9 are loaded first; the rne barriers hide their latency
    actuation_on = nu > 0 and (opt_disableflags & DisableBit.ACTUATION) == 0
    ctrl = float(0.0)
    ctrlrange = wp.vec2(0.0)
    ctrllimited = int(0)
    gain = float(0.0)
    bias = float(0.0)
    forcelimited = int(0)
    forcerange = wp.vec2(0.0)
    if is_act and actuation_on:
      ctrl = ctrl_in[worldid, tid]
      if ((a_flags >> 8) & 1) != 0 and (opt_disableflags & DisableBit.CLAMPCTRL) == 0:
        ctrllimited = 1
        ctrlrange = actuator_ctrlrange[worldid % actuator_ctrlrange.shape[0], tid]
      gainprm = actuator_gainprm[worldid % actuator_gainprm.shape[0], tid]
      gain = gainprm[0]
      if (a_flags & 0xF) == GainType.AFFINE:
        gain = gainprm[0] + gainprm[1] * a_length + gainprm[2] * a_velocity
      if ((a_flags >> 4) & 0xF) == BiasType.AFFINE:
        biasprm = actuator_biasprm[worldid % actuator_biasprm.shape[0], tid]
        bias = biasprm[0] + biasprm[1] * a_length + biasprm[2] * a_velocity
      if ((a_flags >> 9) & 1) != 0:
        forcelimited = 1
        forcerange = actuator_forcerange[worldid % actuator_forcerange.shape[0], tid]
    d_frcrange = wp.vec2(0.0)
    if is_dof and actuation_on and d_frclimited != 0:
      d_frcrange = jnt_actfrcrange[worldid % jnt_actfrcrange.shape[0], d_jnt]

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
      if tid > 0:
        for c in range(tid + 1, subtree_end):
          cfrc += cacc_sh[c]
    if warp == 0:
      part6 = wp.spatial_vector()
      for c in range(lane + 1, nbody, 32):
        part6 += cacc_sh[c]
      for k in range(6):
        part6[k] = _warp_sum(part6[k])
      if tid == 0:
        cfrc += part6
    if is_body:
      # the cvel slot (dead since the traversal) now holds the interaction force
      _st_body_sv(cvel_sh, tid, cfrc)
      if PUBLISH:
        _st_sv(cfrc_int_out, worldid, tid, cfrc)
    _sync()
    if is_dof:
      qfrc_bias_out[worldid, tid] = wp.dot(cdof, cvel_sh[d_body])

    # ---------------------------------------------------------------- P9: actuation
    if not actuation_on:
      if is_act:
        actuator_force_out[worldid, tid] = 0.0
      if is_dof:
        qfrc_actuator_out[worldid, tid] = 0.0
    else:
      if is_act:
        if ctrllimited != 0:
          ctrl = wp.clamp(ctrl, ctrlrange[0], ctrlrange[1])
        force = gain * ctrl + bias
        if forcelimited != 0:
          force = wp.clamp(force, forcerange[0], forcerange[1])
        actuator_force_out[worldid, tid] = force
        # moment of a joint transmission is the gear ratio
        _st_act_f(act_force_sh, tid, force * a_gear)
      _sync()
      if is_dof:
        qfrc = float(0.0)
        for actid in range(nu):
          if act_dof_sh[actid] == tid:
            qfrc += act_force_sh[actid]
        # actuator-level gravity compensation, skip if added as passive force
        if gravity_enabled and actgravcomp != 0:
          qfrc += qfrc_gravcomp
        if d_frclimited != 0:
          qfrc = wp.clamp(qfrc, d_frcrange[0], d_frcrange[1])
        qfrc_actuator_out[worldid, tid] = qfrc

  return kernel


@cache_kernel
def _forward_b_kernel(NB: int, NV: int, NT: int, NSMALL: int, NBIGDOF: int, NBIG: int, PUBLISH: bool, COMPACT_MAPS: bool):
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
    nvtree_max: int,
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
    xfrc_flags_sh = wp.tile_empty(shape=(4,), dtype=int, storage="shared")

    lane = tid & 31
    warp = tid >> 5
    is_body = tid < nbody
    is_dof = tid < nv
    is_tree = tid < ntree

    # ---------------------------------------------------------- Q0: loads, map reset, scratch slots
    xfrc_nz = int(0)
    if is_body:
      xfrc = xfrc_applied_in[worldid, tid]
      _st_body_sv(xfrc_sh, tid, xfrc)
      if not (xfrc == wp.spatial_vector()):
        xfrc_nz = 1
    if lane == 0:
      _st_misc_i(xfrc_flags_sh, warp, _popc(_ballot(xfrc_nz)))
    else:
      _ballot(xfrc_nz)
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
    d_body = int(0)
    d_tree = int(-1)
    d_rowadr = int(0)
    d_rownnz = int(0)
    d_block_adr = int(0)
    d_start = int(0)
    if is_dof:
      d_body = dof_bodyid[tid]
      d_tree = dof_treeid[tid]
      d_rowadr = M_rowadr[tid]
      d_rownnz = M_rownnz[tid]
      d_block_adr = qLD_block_adr[tid]
      d_start = tree_dofadr[d_tree]
    if wp.static(COMPACT_MAPS):
      if is_dof:
        dof_cdof_out[worldid, tid] = -1
      for i in range(tid, nvmax_pad_in, BLOCK):
        cdof_dof_out[worldid, i] = -1
    _sync()
    any_xfrc = (xfrc_flags_sh[0] + xfrc_flags_sh[1] + xfrc_flags_sh[2] + xfrc_flags_sh[3]) != 0
    d_dense = is_dof and d_block_adr != Q_LD_BLOCK_COMPACT
    d_base = int(0)
    d_n = int(1)
    d_i = int(0)
    if d_dense:
      d_base = tree_base_sh[d_tree]
      d_n = tree_dofnum_sh[d_tree]
      d_i = tid - d_start

    # ------------------------------------------------------------ Q1: qfrc_smooth + xfrc_accumulate
    qfrc = float(0.0)
    if is_dof:
      # every dof belongs to a tree (static_eligible), so tree state is indexed unguarded
      if tree_awake_sh[d_tree] != 0:
        qfrc = (
          qfrc_passive_in[worldid, tid]
          - qfrc_bias_in[worldid, tid]
          + qfrc_actuator_in[worldid, tid]
          + qfrc_applied_in[worldid, tid]
        )
      # J^T xfrc_applied over the bodies in this dof's subtree (support._apply_ft); skipped when the
      # world applies no Cartesian force
      if any_xfrc:
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
        # this dof's row of the dense block (zeroed first: branching trees leave structural zeros)
        for c in range(d_n):
          _st_fac(fac_sh, d_base + d_i * d_n + c, 0.0)
        for k in range(d_rownnz):
          col = M_colind[d_rowadr + k] - d_start
          _st_fac(fac_sh, d_base + d_i * d_n + col, M_in[worldid, d_rowadr + k])
        _st_fac(fac_sh, d_base + d_n * d_n + d_i, qfrc)
    _sync()

    # ---------------------------------------------------------- Q2: per-tree dense factor and solve
    # upper Cholesky M = U^T U with the solve, one column per dof thread
    # (smooth._small_cholesky_factorize_solve_block)
    _dense_factor_solve(fac_sh, d_dense, d_base, d_n, d_i, nvtree_max)
    if d_dense:
      # publish the packed upper factor (the unused lower triangle reads as zero) and the solution
      if wp.static(PUBLISH):
        for r in range(d_n):
          value = float(0.0)
          if d_i >= r:
            value = fac_sh[d_base + r * d_n + d_i]
          qLD_out[worldid, d_block_adr + r * d_n + d_i] = value
      x = fac_sh[d_base + d_n * d_n + d_i]
      if tree_awake_sh[d_tree] == 0:
        x = 0.0
      qacc_smooth_out[worldid, tid] = x

    if wp.static(COMPACT_MAPS):
      # ----------------------------------------- Q3: compaction layout (island._compact_dof_layout)
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

      # -------------------------------------------- Q3b: compaction maps (island._map_compact_dofs)
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


@wp.func
def _contact_row(
  worldid: int,
  efcid: int,
  cid: int,
  efc_type: int,
  kbid: wp.vec4,
  pos: float,
  margin: float,
  Jqvel: float,
  efc_type_out: wp.array2d[int],
  efc_id_out: wp.array2d[int],
  efc_pos_out: wp.array2d[float],
  efc_margin_out: wp.array2d[float],
  efc_D_out: wp.array2d[float],
  efc_vel_out: wp.array2d[float],
  efc_aref_out: wp.array2d[float],
  efc_frictionloss_out: wp.array2d[float],
  efc_Jqvel_out: wp.array2d[float],
  publish_sensor_fields: int,
):
  """Row parameters of one contact row from its contact's (k, b, imp, D) (constraint._efc_row).

  ``publish_sensor_fields`` gates the fields only sensors read (:data:`DERIVED_EFC_FIELDS`).
  """
  efc_D_out[worldid, efcid] = kbid[3]
  efc_aref_out[worldid, efcid] = -kbid[0] * kbid[2] * pos - kbid[1] * Jqvel
  if publish_sensor_fields != 0:
    efc_Jqvel_out[worldid, efcid] = Jqvel
    efc_vel_out[worldid, efcid] = Jqvel
    efc_pos_out[worldid, efcid] = pos + margin
    efc_margin_out[worldid, efcid] = margin
  efc_frictionloss_out[worldid, efcid] = 0.0
  efc_type_out[worldid, efcid] = efc_type
  efc_id_out[worldid, efcid] = cid


@wp.func
def _general_row(
  opt_disableflags: int,
  worldid: int,
  timestep: float,
  efcid: int,
  pos_aref: float,
  pos_imp: float,
  invweight: float,
  solref: wp.vec2,
  solimp: vec5,
  margin: float,
  vel: float,
  frictionloss: float,
  type: int,
  id: int,
  type_out: wp.array2d[int],
  id_out: wp.array2d[int],
  pos_out: wp.array2d[float],
  margin_out: wp.array2d[float],
  D_out: wp.array2d[float],
  vel_out: wp.array2d[float],
  aref_out: wp.array2d[float],
  frictionloss_out: wp.array2d[float],
  publish_sensor_fields: int,
):
  """constraint._efc_row with the sensor-only fields (:data:`DERIVED_EFC_FIELDS`) behind a flag."""
  kbid = constraint._efc_kbid(opt_disableflags, timestep, pos_imp, invweight, solref, solimp)
  D_out[worldid, efcid] = kbid[3]
  aref_out[worldid, efcid] = -kbid[0] * kbid[2] * pos_aref - kbid[1] * vel
  if publish_sensor_fields != 0:
    vel_out[worldid, efcid] = vel
    pos_out[worldid, efcid] = pos_aref + margin
    margin_out[worldid, efcid] = margin
  frictionloss_out[worldid, efcid] = frictionloss
  type_out[worldid, efcid] = type
  id_out[worldid, efcid] = id


# ------------------------------------------------------------------------------------------------
# Per-world contact buckets: Newton assigns contact ids with a global atomic counter, so the
# contacts of one world are scattered over [0, nacon). One pass over the live contacts appends every
# constraint-eligible penetrating id (the predicate of constraint._efc_contact_init: the other
# contacts build no rows) to its world's bucket. The bucket capacity is naconmax (exact: any
# distribution of the global pool fits) while nworld * capacity stays within _GROUP_BUDGET_BYTES,
# otherwise the budget divided by nworld (at least the nconmax share); a world with more active
# contacts keeps the first ``capacity`` and raises OverflowType.NARROWPHASE, which is stricter than
# the stock global pool. Contacts may be appended every substep (wake injection), so the pass runs
# per substep; forward_a zeroes the per-world counts first. The slot order within a bucket follows
# the atomics, i.e. the contact row order is unspecified exactly as with the stock atomic row
# allocation.
# ------------------------------------------------------------------------------------------------

# fixed grid for the bucket pass (it grid-strides over nacon, which is a device scalar)
_GROUP_THREADS = 65536
# memory budget of the per-world contact buckets (ids): 16 MB is 4096 ids per world at 1024 worlds
_GROUP_BUDGET_BYTES = 16 << 20


@wp.kernel(enable_backward=False, grid_stride=False)
def _world_contact_bucket(
  # Data in:
  nacon_in: wp.array[int],
  contact_worldid_in: wp.array[int],
  contact_type_in: wp.array[int],
  contact_dist_in: wp.array[float],
  contact_includemargin_in: wp.array[float],
  # In:
  capacity: int,
  total_threads: int,
  # Out:
  count_out: wp.array[int],
  ids_out: wp.array[int],
  overflow_out: wp.array[int],
):
  tid = wp.tid()
  n = wp.min(nacon_in[0], contact_worldid_in.shape[0])
  for cid in range(tid, n, total_threads):
    if (contact_type_in[cid] & ContactType.CONSTRAINT) == 0:
      continue
    # inactive (non-penetrating) contacts own no rows; written as in constraint._efc_contact_init
    if not (contact_dist_in[cid] - contact_includemargin_in[cid] < 0.0):
      continue
    worldid = contact_worldid_in[cid]
    slot = wp.atomic_add(count_out, worldid, 1)
    if slot < capacity:
      ids_out[worldid * capacity + slot] = cid
    else:
      wp.atomic_or(overflow_out, worldid, OverflowType.NARROWPHASE)


@dataclasses.dataclass
class ContactGroups:
  """Per-world contact id buckets consumed by ``forward_m``.

  Attributes:
    capacity: ids per world.
    count: contacts per world, zeroed by ``forward_a``          (nworld,)
    ids: contact ids of world w at [w * capacity, w * capacity + count[w]) (nworld * capacity,)
  """

  capacity: int
  count: wp.array
  ids: wp.array


def bucket_capacity(nworld: int, naconmax: int) -> int:
  """Contact ids per world bucket: naconmax (exact) within the budget, else the budget share."""
  share = max(1, -(-naconmax // nworld))
  budget = max(share, (_GROUP_BUDGET_BYTES // 4) // max(nworld, 1))
  return max(1, min(naconmax, budget))


def contact_groups(d: Data) -> ContactGroups:
  """Allocate the per-world contact buckets for one fused forward pass."""
  capacity = bucket_capacity(d.nworld, d.naconmax)
  return ContactGroups(capacity, wp.empty((d.nworld,), dtype=int), wp.empty((d.nworld * capacity,), dtype=int))


def _bucket_contacts(m: Model, d: Data, groups: ContactGroups):
  """Append every active (row-building) contact id to its world's bucket (counts start at zero)."""
  if m.opt.disableflags & DisableBit.CONTACT:
    return
  threads = min(_GROUP_THREADS, max(d.naconmax, 1))
  wp.launch(
    _world_contact_bucket,
    dim=threads,
    inputs=[d.nacon, d.contact.worldid, d.contact.type, d.contact.dist, d.contact.includemargin, groups.capacity, threads],
    outputs=[groups.count, groups.ids, d.overflow],
  )


# resident CTAs per SM requested from the compiler for the register-bound middle kernel: 7 CTAs/SM
# hold 1024 worlds in one wave on 170 SMs (<= 73 registers; the small spill is L1-resident)
_FORWARD_M_MIN_BLOCKS = 7


@cache_kernel
def _forward_m_kernel(NB: int, NV: int, NT: int, PUBLISH: bool):
  BLOCK = NV

  @wp.kernel(module="unique", enable_backward=False, grid_stride=False, launch_bounds=(BLOCK, _FORWARD_M_MIN_BLOCKS))
  def kernel(
    # Model:
    nv: int,
    nbody: int,
    ntree: int,
    neq: int,
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
    contact_pos_in: wp.array[wp.vec3],
    contact_frame_in: wp.array2d[wp.vec3],
    contact_friction_in: wp.array[vec5],
    contact_solref_in: wp.array[wp.vec2],
    contact_solimp_in: wp.array[vec5],
    world_con_capacity_in: int,
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
    tree_dofnum_sh = wp.tile_empty(shape=(NT,), dtype=int, storage="shared")
    edge_sh = wp.tile_empty(shape=(NT,), dtype=wp.uint32, storage="shared")
    # M1 stashes the equality wake data here before the sleep publish fills it with body states
    body_awake_sh = wp.tile_empty(shape=(NB,), dtype=int, storage="shared")
    # per-body tables for the contact rows: weld body, and dofadr + 1 | dofnum << 8 | rootid << 16
    bweld_sh = wp.tile_empty(shape=(NB,), dtype=int, storage="shared")
    binfo_sh = wp.tile_empty(shape=(NB,), dtype=int, storage="shared")
    misc_sh = wp.tile_empty(shape=(16,), dtype=int, storage="shared")
    scan_sh = wp.tile_empty(shape=(8,), dtype=int, storage="shared")
    # per-dof tables: parent dof for the merged ancestor chains, velocity for Jqvel
    dof_parent_sh = wp.tile_empty(shape=(NV,), dtype=int, storage="shared")
    qvel_sh = wp.tile_empty(shape=(NV,), dtype=float, storage="shared")
    # per-chunk contact stash for the column and row phases
    c_cid_sh = wp.tile_empty(shape=(NV,), dtype=int, storage="shared")
    c_row_sh = wp.tile_empty(shape=(NV,), dtype=int, storage="shared")
    c_rowadr_sh = wp.tile_empty(shape=(NV,), dtype=int, storage="shared")
    c_rownnz_sh = wp.tile_empty(shape=(NV,), dtype=int, storage="shared")
    c_body1_sh = wp.tile_empty(shape=(NV,), dtype=int, storage="shared")
    c_body2_sh = wp.tile_empty(shape=(NV,), dtype=int, storage="shared")
    c_col_sh = wp.tile_empty(shape=(NV,), dtype=int, storage="shared")
    c_type_sh = wp.tile_empty(shape=(NV,), dtype=int, storage="shared")  # type | ndim << 8
    c_k_sh = wp.tile_empty(shape=(NV,), dtype=float, storage="shared")
    c_b_sh = wp.tile_empty(shape=(NV,), dtype=float, storage="shared")
    c_imp_sh = wp.tile_empty(shape=(NV,), dtype=float, storage="shared")
    c_D_sh = wp.tile_empty(shape=(NV,), dtype=float, storage="shared")
    c_pos_sh = wp.tile_empty(shape=(NV,), dtype=float, storage="shared")
    c_margin_sh = wp.tile_empty(shape=(NV,), dtype=float, storage="shared")

    lane = tid & 31
    warp = tid >> 5
    timestep = opt_timestep[worldid % opt_timestep.shape[0]]
    impratio_invsqrt = opt_impratio_invsqrt[worldid % opt_impratio_invsqrt.shape[0]]

    constraint_on = (opt_disableflags & DisableBit.CONSTRAINT) == 0
    eq_on = constraint_on and (opt_disableflags & DisableBit.EQUALITY) == 0
    fric_on = constraint_on and (opt_disableflags & DisableBit.FRICTIONLOSS) == 0
    lim_on = constraint_on and (opt_disableflags & DisableBit.LIMIT) == 0
    con_on = constraint_on and (opt_disableflags & DisableBit.CONTACT) == 0

    # ------------------------------------------------------------------ M0: prefetch and tables
    # The dependent lookup chains of the later phases are software-pipelined: the first rounds run
    # here (in parallel across chains), the last rounds are issued one phase before their use so
    # that no phase waits on a chain by itself.
    if tid < NT:
      _st_tree32_u(edge_sh, tid, wp.uint32(0))
    if tid < ntree:
      _st_tree32_i(tree_asleep_sh, tid, tree_asleep_out[worldid, tid])
      _st_tree32_i(tree_dofnum_sh, tid, tree_dofnum[tid])
    b_tree = int(-1)
    b_root = int(0)
    if tid < nbody:
      b_tree = body_treeid[tid]
      b_root = body_rootid[tid]
      _st_body64_i(bweld_sh, tid, body_weldid[tid])
      _st_body64_i(binfo_sh, tid, (body_dofadr[tid] + 1) | (body_dofnum[tid] << 8) | (b_root << 16))
    d_body = int(0)
    d_tree = int(-1)
    frictionloss = float(0.0)
    if tid < nv:
      d_body = dof_bodyid[tid]
      d_tree = dof_treeid[tid]
      _st_chunk_i(dof_parent_sh, tid, dof_parentid[tid])
      _st_chunk_f(qvel_sh, tid, qvel_in[worldid, tid])
      if fric_on:
        frictionloss = dof_frictionloss[worldid % dof_frictionloss.shape[0], tid]
    # equalities (all JOINT, so eq_jnt_adr is the identity): the row predicate and the wake chain
    # share the first rounds
    e_active = int(0)
    e_id1 = int(-1)
    e_id2 = int(-1)
    e_dofadr1 = int(0)
    e_dofadr2 = int(-1)
    e_qposadr1 = int(0)
    e_qposadr2 = int(0)
    e_tree1 = int(-1)
    e_tree2 = int(-1)
    if tid < neq:
      if eq_active_in[worldid, tid]:
        e_active = 1
        e_id1 = eq_obj1id[tid]
        e_id2 = eq_obj2id[tid]
        if e_id1 >= 0:
          e_dofadr1 = jnt_dofadr[e_id1]
          e_qposadr1 = jnt_qposadr[e_id1]
          e_tree1 = body_treeid[jnt_bodyid[e_id1]]
        if e_id2 >= 0:
          e_dofadr2 = jnt_dofadr[e_id2]
          e_qposadr2 = jnt_qposadr[e_id2]
          e_tree2 = body_treeid[jnt_bodyid[e_id2]]
    e_rows = int(0)
    e_nnz = int(0)
    if eq_on and e_active == 1:
      e_rows = 1
      e_nnz = 1
      if e_id2 > -1:
        e_nnz = 2
    # dof friction: one row per dof with frictionloss > 0
    f_rows = int(0)
    if fric_on and tid < nv and frictionloss > 0.0:
      f_rows = 1
    # slide/hinge limits: one row per violated limit
    l_rows = int(0)
    l_jntid = int(-1)
    l_dofadr = int(0)
    l_pos = float(0.0)
    l_dist_min = float(0.0)
    l_dist_max = float(0.0)
    l_margin = float(0.0)
    if lim_on and tid < nlimit:
      l_jntid = jnt_limited_slide_hinge_adr[tid]
      jntrange = jnt_range[worldid % jnt_range.shape[0], l_jntid]
      l_dofadr = jnt_dofadr[l_jntid]
      l_margin = jnt_margin[worldid % jnt_margin.shape[0], l_jntid]
      qpos = qpos_in[worldid, jnt_qposadr[l_jntid]]
      l_dist_min = qpos - jntrange[0]
      l_dist_max = jntrange[1] - qpos
      l_pos = wp.min(l_dist_min, l_dist_max) - l_margin
      if l_pos < 0.0:
        l_rows = 1
    # first contact of this thread (the bucket holds active contacts only); the geom bodies follow
    # after the scan
    ncon = int(0)
    cstart = worldid * world_con_capacity_in
    if con_on:
      ncon = wp.min(world_con_count_in[worldid], world_con_capacity_in)
    p_cid = int(-1)
    p_dist = float(0.0)
    p_includemargin = float(0.0)
    p_condim = int(0)
    p_geom = wp.vec2i(0, 0)
    p_gb1 = int(0)
    p_gb2 = int(0)
    if tid < ncon:
      p_cid = world_con_list_in[cstart + tid]
      p_dist = contact_dist_in[p_cid]
      p_includemargin = contact_includemargin_in[p_cid]
      p_condim = contact_dim_in[p_cid]
      p_geom = contact_geom_in[p_cid]
    _sync()

    # ------------------------------------------ M3a: equality, friction and limit rows (one scan)
    # canonical order ne | nf | nl | contacts; within a family, rows follow the item order. Row and
    # J-nonzero addresses are exclusive prefixes (stock: atomic allocation). The three families fit
    # one block each (neq, nv, nlimit <= BLOCK), so one packed scan allocates all of them: rows_eq |
    # rows_f << 8 | rows_l << 16 | nnz_eq << 24 (every field <= BLOCK < 256).
    packed = e_rows | (f_rows << 8) | (l_rows << 16) | (e_nnz << 24)
    excl = _block_scan(scan_sh, packed, lane, warp)
    total = scan_sh[4]
    ne = total & 0xFF
    nf = (total >> 8) & 0xFF
    nl = (total >> 16) & 0xFF
    e_efcid = excl & 0xFF
    f_efcid = ne + ((excl >> 8) & 0xFF)
    l_efcid = ne + nf + ((excl >> 16) & 0xFF)
    e_rowadr = (excl >> 24) & 0xFF
    f_rowadr = ((total >> 24) & 0xFF) + ((excl >> 8) & 0xFF)
    l_rowadr = ((total >> 24) & 0xFF) + nf + ((excl >> 16) & 0xFF)
    nnz_base = ((total >> 24) & 0xFF) + nf + nl
    if ne + nf + nl > njmax_in:
      # rows at or beyond njmax are counted in nefc but not written and own no nonzeros (as in
      # stock): redo the nonzero allocation with that cutoff
      e_nnz_ok = e_nnz
      if e_efcid >= njmax_in:
        e_nnz_ok = 0
      f_nnz_ok = f_rows
      if f_efcid >= njmax_in:
        f_nnz_ok = 0
      l_nnz_ok = l_rows
      if l_efcid >= njmax_in:
        l_nnz_ok = 0
      excl2 = _block_scan(scan_sh, e_nnz_ok | (f_nnz_ok << 8) | (l_nnz_ok << 16), lane, warp)
      total2 = scan_sh[4]
      e_rowadr = excl2 & 0xFF
      f_rowadr = (total2 & 0xFF) + ((excl2 >> 8) & 0xFF)
      l_rowadr = (total2 & 0xFF) + ((total2 >> 8) & 0xFF) + ((excl2 >> 16) & 0xFF)
      nnz_base = (total2 & 0xFF) + ((total2 >> 8) & 0xFF) + ((total2 >> 16) & 0xFF)
    # last rounds of the pipelined chains: the wake states of the equality trees and the geom
    # bodies of the first contact chunk (consumed after the row writes below)
    e_s1 = int(SleepState.STATIC)
    e_s2 = int(SleepState.STATIC)
    if e_tree1 >= 0:
      e_s1 = tree_awake_in[worldid, e_tree1]
    if e_tree2 >= 0:
      e_s2 = tree_awake_in[worldid, e_tree2]
    if tid < ncon:
      p_gb1 = geom_bodyid[p_geom[0]]
      p_gb2 = geom_bodyid[p_geom[1]]

    if e_rows == 1 and e_efcid < njmax_in:
      # every row so far is a one-row block below njmax, so the block index is the row index
      efc_jtdaj_adr_out[worldid, e_efcid] = e_efcid
      efc_jtdaj_nrow_out[worldid, e_efcid] = 1
      data = eq_data[worldid % eq_data.shape[0], tid]
      qpos0_id = worldid % qpos0.shape[0]
      dof_invweight0_id = worldid % dof_invweight0.shape[0]
      deriv_2 = float(0.0)
      if e_id2 > -1:
        dif = qpos_in[worldid, e_qposadr2] - qpos0[qpos0_id, e_qposadr2]
        # Horner's method for polynomials
        rhs = data[0] + dif * (data[1] + dif * (data[2] + dif * (data[3] + dif * data[4])))
        deriv_2 = data[1] + dif * (2.0 * data[2] + dif * (3.0 * data[3] + dif * 4.0 * data[4]))
        pos = qpos_in[worldid, e_qposadr1] - qpos0[qpos0_id, e_qposadr1] - rhs
        Jqvel = qvel_sh[e_dofadr1] - qvel_sh[e_dofadr2] * deriv_2
        invweight = dof_invweight0[dof_invweight0_id, e_dofadr1] + dof_invweight0[dof_invweight0_id, e_dofadr2]
      else:
        pos = qpos_in[worldid, e_qposadr1] - qpos0[qpos0_id, e_qposadr1] - data[0]
        Jqvel = qvel_sh[e_dofadr1]
        invweight = dof_invweight0[dof_invweight0_id, e_dofadr1]

      if e_rowadr + e_nnz > njmax_nnz_in:
        # stock leaves this row half-written; publish an empty Jacobian row and flag the overflow
        efc_J_rownnz_out[worldid, e_efcid] = 0
        efc_J_rowadr_out[worldid, e_efcid] = 0
        wp.atomic_or(overflow_out, worldid, OverflowType.NJMAX_NNZ)
      else:
        efc_J_rownnz_out[worldid, e_efcid] = e_nnz
        efc_J_rowadr_out[worldid, e_efcid] = e_rowadr
        efc_J_colind_out[worldid, 0, e_rowadr] = e_dofadr1
        efc_J_out[worldid, 0, e_rowadr] = 1.0
        tree_a = dof_treeid[e_dofadr1]
        tree_b = int(-1)
        if e_id2 > -1:
          efc_J_colind_out[worldid, 0, e_rowadr + 1] = e_dofadr2
          efc_J_out[worldid, 0, e_rowadr + 1] = -deriv_2
          tree_b = dof_treeid[e_dofadr2]
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

      _general_row(
        opt_disableflags,
        worldid,
        timestep,
        e_efcid,
        pos,
        pos,
        invweight,
        eq_solref[worldid % eq_solref.shape[0], tid],
        eq_solimp[worldid % eq_solimp.shape[0], tid],
        0.0,
        Jqvel,
        0.0,
        ConstraintType.EQUALITY,
        tid,
        efc_type_out,
        efc_id_out,
        efc_pos_out,
        efc_margin_out,
        efc_D_out,
        efc_vel_out,
        efc_aref_out,
        efc_frictionloss_out,
        wp.static(int(PUBLISH)),
      )

    if f_rows == 1 and f_efcid < njmax_in:
      efc_jtdaj_adr_out[worldid, f_efcid] = f_efcid
      efc_jtdaj_nrow_out[worldid, f_efcid] = 1
      if f_rowadr + 1 > njmax_nnz_in:
        efc_J_rownnz_out[worldid, f_efcid] = 0
        efc_J_rowadr_out[worldid, f_efcid] = 0
        wp.atomic_or(overflow_out, worldid, OverflowType.NJMAX_NNZ)
      else:
        efc_J_rownnz_out[worldid, f_efcid] = 1
        efc_J_rowadr_out[worldid, f_efcid] = f_rowadr
        efc_J_colind_out[worldid, 0, f_rowadr] = tid
        efc_J_out[worldid, 0, f_rowadr] = 1.0
      _mark_tree_edge(edge_sh, d_tree, -1)
      _general_row(
        opt_disableflags,
        worldid,
        timestep,
        f_efcid,
        0.0,
        0.0,
        dof_invweight0[worldid % dof_invweight0.shape[0], tid],
        dof_solref[worldid % dof_solref.shape[0], tid],
        dof_solimp[worldid % dof_solimp.shape[0], tid],
        0.0,
        qvel_sh[tid],
        frictionloss,
        ConstraintType.FRICTION_DOF,
        tid,
        efc_type_out,
        efc_id_out,
        efc_pos_out,
        efc_margin_out,
        efc_D_out,
        efc_vel_out,
        efc_aref_out,
        efc_frictionloss_out,
        wp.static(int(PUBLISH)),
      )

    if l_rows == 1 and l_efcid < njmax_in:
      efc_jtdaj_adr_out[worldid, l_efcid] = l_efcid
      efc_jtdaj_nrow_out[worldid, l_efcid] = 1
      J = float(l_dist_min < l_dist_max) * 2.0 - 1.0
      if l_rowadr + 1 > njmax_nnz_in:
        efc_J_rownnz_out[worldid, l_efcid] = 0
        efc_J_rowadr_out[worldid, l_efcid] = 0
        wp.atomic_or(overflow_out, worldid, OverflowType.NJMAX_NNZ)
      else:
        efc_J_rownnz_out[worldid, l_efcid] = 1
        efc_J_rowadr_out[worldid, l_efcid] = l_rowadr
        efc_J_colind_out[worldid, 0, l_rowadr] = l_dofadr
        efc_J_out[worldid, 0, l_rowadr] = J
      _mark_tree_edge(edge_sh, dof_treeid[l_dofadr], -1)
      _general_row(
        opt_disableflags,
        worldid,
        timestep,
        l_efcid,
        l_pos,
        l_pos,
        dof_invweight0[worldid % dof_invweight0.shape[0], l_dofadr],
        jnt_solref[worldid % jnt_solref.shape[0], l_jntid],
        jnt_solimp[worldid % jnt_solimp.shape[0], l_jntid],
        l_margin,
        J * qvel_sh[l_dofadr],
        0.0,
        ConstraintType.LIMIT_JOINT,
        l_jntid,
        efc_type_out,
        efc_id_out,
        efc_pos_out,
        efc_margin_out,
        efc_D_out,
        efc_vel_out,
        efc_aref_out,
        efc_frictionloss_out,
        wp.static(int(PUBLISH)),
      )
    row_base = ne + nf + nl

    # ------------------------------------------------------------------- M3b: contact rows
    # pyramidal contacts of this world: ndim rows per contact (the bucket pass keeps the
    # constraint-eligible penetrating contacts only), one jtdaj block per contact. Per chunk of
    # BLOCK contacts: one thread per contact allocates rows/nonzeros/blocks, writes the bookkeeping
    # and evaluates the impedance once (it depends on the contact, not on the row); the Jacobian
    # entries are then computed one (contact, ancestor dof) column per thread for all of the
    # contact's rows, and the row parameters follow from a segmented warp reduction of J * qvel over
    # the contact's columns, so the written entries are never re-read. The next chunk's contact
    # records are loaded while the current chunk is processed.
    nblock = wp.min(row_base, njmax_in)
    for i0 in range(0, ncon, BLOCK):
      i = i0 + tid
      chunk_n = wp.min(BLOCK, ncon - i0)
      cid = p_cid
      active = int(0)
      rows = int(0)
      nnz = int(0)
      ndim = int(0)
      rownnz = int(0)
      body1 = int(0)
      body2 = int(0)
      gb1 = p_gb1
      gb2 = p_gb2
      pos = p_dist - p_includemargin
      includemargin = p_includemargin
      if i < ncon:
        active = 1
        ndim = 1
        if p_condim > 1:
          ndim = 2 * (p_condim - 1)
        rows = ndim
        body1 = bweld_sh[gb1]
        body2 = bweld_sh[gb2]
        # count the merged ancestor chain excluding common dofs (constraint._efc_contact_init)
        info1 = binfo_sh[body1]
        info2 = binfo_sh[body2]
        da1 = ((info1 & 0xFF) - 1) + ((info1 >> 8) & 0xFF) - 1
        da2 = ((info2 & 0xFF) - 1) + ((info2 >> 8) & 0xFF) - 1
        while da1 >= 0 or da2 >= 0:
          da = wp.max(da1, da2)
          if da1 == da and da2 == da:
            break
          if da1 == da:
            da1 = dof_parent_sh[da1]
          if da2 == da:
            da2 = dof_parent_sh[da2]
          rownnz += 1
        nnz = rownnz * ndim
      # rows and blocks in one scan; the chunk sums bound the fields: rows <= 10 * BLOCK (11 bits),
      # blocks <= BLOCK (8 bits). Blocks assume every row of the chunk fits below njmax, which the
      # rare overflow path below corrects
      excl = _block_scan(scan_sh, rows | (active << 11), lane, warp)
      row_excl = excl & 0x7FF
      base = row_base + row_excl
      rows_total = scan_sh[4] & 0x7FF
      blocks = active
      block_excl = excl >> 11
      blocks_total = scan_sh[4] >> 11
      if row_base + rows_total > njmax_in:
        blocks = 0
        if active == 1 and base < njmax_in:
          blocks = 1
        block_excl = _block_scan(scan_sh, blocks, lane, warp)
        blocks_total = scan_sh[4]
      jgid = nblock + block_excl
      rowadr = nnz_base + _block_scan(scan_sh, nnz, lane, warp)
      nnz_total = scan_sh[4]
      nnz_ok = int(1)
      if rowadr + nnz > njmax_nnz_in:
        nnz_ok = 0

      efc_type = int(ConstraintType.CONTACT_FRICTIONLESS)
      kbid = wp.vec4(0.0)
      if active == 1:
        for dim in range(ndim):
          efcid = base + dim
          if efcid >= njmax_in:
            contact_efc_address_out[cid, dim] = -1
          else:
            contact_efc_address_out[cid, dim] = efcid
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
          _mark_tree_edge(edge_sh, body_treeid[gb1], body_treeid[gb2])
        # row parameters shared by the contact's rows (constraint._efc_contact_update); the
        # inverse weight uses the geom bodies
        body_invweight0_id = worldid % body_invweight0.shape[0]
        invweight = body_invweight0[body_invweight0_id, gb1][0] + body_invweight0[body_invweight0_id, gb2][0]
        if ndim > 1:
          fri0 = contact_friction_in[cid][0]
          invweight = invweight + fri0 * fri0 * invweight
          invweight = invweight * 2.0 * fri0 * fri0 * impratio_invsqrt * impratio_invsqrt
          efc_type = int(ConstraintType.CONTACT_PYRAMIDAL)
        kbid = constraint._efc_kbid(opt_disableflags, timestep, pos, invweight, contact_solref_in[cid], contact_solimp_in[cid])
        # contacts without Jacobian entries (both bodies static, or entries beyond njmax_nnz) have
        # no column thread: their rows carry Jqvel = 0
        if rownnz * nnz_ok == 0:
          for dim in range(ndim):
            efcid = base + dim
            if efcid < njmax_in:
              _contact_row(
                worldid,
                efcid,
                cid,
                efc_type,
                kbid,
                pos,
                includemargin,
                0.0,
                efc_type_out,
                efc_id_out,
                efc_pos_out,
                efc_margin_out,
                efc_D_out,
                efc_vel_out,
                efc_aref_out,
                efc_frictionloss_out,
                efc_Jqvel_out,
                wp.static(int(PUBLISH)),
              )
      else:
        cid = -1
      # stash the chunk's contacts for the column phase; contacts whose nonzeros do not fit own no
      # entries (rownnz stashed as 0)
      _st_chunk_i(c_cid_sh, tid, cid)
      _st_chunk_i(c_row_sh, tid, row_excl)
      _st_chunk_i(c_rowadr_sh, tid, rowadr)
      _st_chunk_i(c_rownnz_sh, tid, rownnz * nnz_ok)
      _st_chunk_i(c_body1_sh, tid, body1)
      _st_chunk_i(c_body2_sh, tid, body2)
      _st_chunk_i(c_type_sh, tid, efc_type | (ndim << 8))
      _st_chunk_f(c_k_sh, tid, kbid[0])
      _st_chunk_f(c_b_sh, tid, kbid[1])
      _st_chunk_f(c_imp_sh, tid, kbid[2])
      _st_chunk_f(c_D_sh, tid, kbid[3])
      _st_chunk_f(c_pos_sh, tid, pos)
      _st_chunk_f(c_margin_sh, tid, includemargin)
      _sync()
      # next chunk, first round (the later rounds follow the packing and column phases)
      i_next = i0 + BLOCK + tid
      p_cid = -1
      if i_next < ncon:
        p_cid = world_con_list_in[cstart + i_next]
      # column slots: contact c owns [c_col[c], c_col[c] + rownnz) packed so that no contact
      # straddles a warp (rownnz <= 2 * NVTREE_CAP = 32), which lets the segmented warp reduction
      # below sum a contact's columns without cross-warp carries; the total lands in scan_sh[5]
      if tid == 0:
        col = int(0)
        for c in range(chunk_n):
          n = c_rownnz_sh[c]
          if (col & 31) + n > 32:
            col = (col | 31) + 1
          _st_chunk_i(c_col_sh, c, col)
          col += n
        _st_scan_i(scan_sh, 5, col)
      _sync()
      cols_total = scan_sh[5]
      # next chunk, second round
      if i_next < ncon:
        p_dist = contact_dist_in[p_cid]
        p_includemargin = contact_includemargin_in[p_cid]
        p_condim = contact_dim_in[p_cid]
        p_geom = contact_geom_in[p_cid]

      # Jacobian entries, one (contact, ancestor dof) column per thread for all ndim rows
      # (constraint._efc_contact_jac_sparse; the per-row values only differ in the friction term),
      # then the contact's Jqvel per row by a segmented sum towards its first column, whose thread
      # writes the row parameters (_efc_row)
      for c0 in range(0, cols_total, BLOCK):
        col = c0 + tid
        seg = int(-1)
        lo = int(0)
        k = int(0)
        ndim = int(0)
        efcid0 = int(0)
        Jq = vec10()
        if col < cols_total:
          # owning contact: the last chunk entry whose column slot is <= col (contacts without
          # columns share the slot of their successor, so the search never lands on one); columns
          # in the padding after a contact's entries stay idle
          hi = chunk_n - 1
          while lo < hi:
            mid = (lo + hi + 1) // 2
            if c_col_sh[mid] <= col:
              lo = mid
            else:
              hi = mid - 1
          k = col - c_col_sh[lo]
          if k < c_rownnz_sh[lo]:
            seg = lo
        if seg >= 0:
          rownnz = c_rownnz_sh[lo]
          cid = c_cid_sh[lo]
          body1 = c_body1_sh[lo]
          body2 = c_body2_sh[lo]
          type_ndim = c_type_sh[lo]
          pyramidal = (type_ndim & 0xFF) == ConstraintType.CONTACT_PYRAMIDAL
          ndim = type_ndim >> 8
          efcid0 = row_base + c_row_sh[lo]
          # the contact records and both root coms are issued before the ancestor walk so that
          # their latency overlaps it
          con_pos = contact_pos_in[cid]
          frame_0 = contact_frame_in[cid, 0]
          frame_1 = wp.vec3(0.0)
          frame_2 = wp.vec3(0.0)
          friction = vec5()
          if pyramidal:
            frame_1 = contact_frame_in[cid, 1]
            frame_2 = contact_frame_in[cid, 2]
            friction = contact_friction_in[cid]
          info1 = binfo_sh[body1]
          info2 = binfo_sh[body2]
          com1 = subtree_com_in[worldid, info1 >> 16]
          com2 = subtree_com_in[worldid, info2 >> 16]
          # k-th dof of the merged ancestor chain; common ancestors are excluded from rownnz, so
          # one body owns it
          da1 = ((info1 & 0xFF) - 1) + ((info1 >> 8) & 0xFF) - 1
          da2 = ((info2 & 0xFF) - 1) + ((info2 >> 8) & 0xFF) - 1
          da = wp.max(da1, da2)
          for _step in range(k):
            if da1 == da:
              da1 = dof_parent_sh[da1]
            if da2 == da:
              da2 = dof_parent_sh[da2]
            da = wp.max(da1, da2)
          offset = con_pos - com2
          sign = float(1.0)
          if da1 == da:
            offset = con_pos - com1
            sign = -1.0
          cdof = cdof_in[worldid, da]
          cdof_ang = wp.spatial_top(cdof)
          jacp_dif = (wp.spatial_bottom(cdof) + wp.cross(cdof_ang, offset)) * sign
          jacr_dif = cdof_ang * sign
          qvel_da = qvel_sh[da]
          adr0 = c_rowadr_sh[lo] + k
          for dim in range(10):
            if dim < ndim:
              frame_i = wp.vec3(0.0)
              dimid2 = int(0)
              frii = float(0.0)
              if pyramidal:
                dimid2 = dim / 2 + 1
                frii = friction[dimid2 - 1]
                if dimid2 == 1:
                  frame_i = frame_1
                elif dimid2 == 2:
                  frame_i = frame_2
                elif dimid2 == 3:
                  frame_i = frame_0
                elif dimid2 == 4:
                  frame_i = frame_1
                else:
                  frame_i = frame_2
              J = float(0.0)
              Ji = float(0.0)
              for xyz in range(3):
                J += frame_0[xyz] * jacp_dif[xyz]
                if pyramidal:
                  if dimid2 < 3:
                    Ji += frame_i[xyz] * jacp_dif[xyz]
                  else:
                    Ji += frame_i[xyz] * jacr_dif[xyz]
              if pyramidal:
                if dim % 2 == 0:
                  J += Ji * frii
                else:
                  J -= Ji * frii
              Jq[dim] = J * qvel_da
              if efcid0 + dim < njmax_in:
                adr = adr0 + dim * rownnz
                efc_J_colind_out[worldid, 0, adr] = da
                efc_J_out[worldid, 0, adr] = J
        # every lane of the block takes part in the warp reduction (warp-uniform row count)
        ndim_warp = _warp_max(ndim)
        Jq = _segment_sum_head10(Jq, seg, ndim_warp)
        if seg >= 0 and k == 0:
          kbid = wp.vec4(c_k_sh[lo], c_b_sh[lo], c_imp_sh[lo], c_D_sh[lo])
          pos = c_pos_sh[lo]
          margin = c_margin_sh[lo]
          efc_type = c_type_sh[lo] & 0xFF
          for dim in range(10):
            if dim < ndim:
              efcid = efcid0 + dim
              if efcid < njmax_in:
                _contact_row(
                  worldid,
                  efcid,
                  cid,
                  efc_type,
                  kbid,
                  pos,
                  margin,
                  Jq[dim],
                  efc_type_out,
                  efc_id_out,
                  efc_pos_out,
                  efc_margin_out,
                  efc_D_out,
                  efc_vel_out,
                  efc_aref_out,
                  efc_frictionloss_out,
                  efc_Jqvel_out,
                  wp.static(int(PUBLISH)),
                )
      _sync()
      # next chunk, third round
      if i_next < ncon:
        p_gb1 = geom_bodyid[p_geom[0]]
        p_gb2 = geom_bodyid[p_geom[1]]
      row_base += rows_total
      nnz_base += nnz_total
      nblock += blocks_total

    if tid == 0:
      ne_out[worldid] = ne
      nf_out[worldid] = nf
      nl_out[worldid] = nl
      nefc_out[worldid] = row_base
      efc_jtdaj_nblock_out[worldid] = nblock
    # equality wake stash (neq <= NB, see static_eligible); the sleep states are offset by one so
    # that STATIC (-1) packs:
    # active | (s1 + 1) << 1 | (s2 + 1) << 3 | (tree1 + 1) << 5 | (tree2 + 1) << 12
    if tid < neq:
      packed_eq = e_active | ((e_s1 + 1) << 1) | ((e_s2 + 1) << 3) | ((e_tree1 + 1) << 5) | ((e_tree2 + 1) << 12)
      _st_body64_i(body_awake_sh, tid, packed_eq)
    _sync()

    # -------------------------------------------------------------- M1: sleep.wake_equality (JOINT)
    # one thread walks the equalities in order (stock: one thread per equality with benign races);
    # the awake test uses the pre-wake tree_awake snapshot exactly like the stock kernel
    if tid == 0:
      for eqid in range(neq):
        packed_eq = body_awake_sh[eqid]
        if (packed_eq & 1) == 0:
          continue
        s1 = ((packed_eq >> 1) & 0x3) - 1
        s2 = ((packed_eq >> 3) & 0x3) - 1
        tree1 = ((packed_eq >> 5) & 0x7F) - 1
        tree2 = ((packed_eq >> 12) & 0x7F) - 1
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
    b_mocap_root = int(-1)
    if tid < nbody:
      b_mocap_root = body_mocapid[b_root]
    _publish_sleep_state(
      tree_asleep_sh,
      body_awake_sh,
      misc_sh,
      nbody,
      nv,
      ntree,
      b_tree,
      b_mocap_root,
      d_body,
      d_tree,
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

    # ----------------------------------------------------- M4: islands (island._flood_fill_bitsets)
    # one thread per tree: the component of a tree is the transitive closure of its edge bitset;
    # islands are numbered by their smallest tree, exactly like the serial flood fill
    reach = wp.uint32(0)
    has_edge = int(0)
    root = int(0)
    if tid < ntree:
      reach = edge_sh[tid]
      if reach != wp.uint32(0):
        has_edge = 1
        reach |= wp.uint32(1) << wp.uint32(tid)
        for _it in range(ntree):
          grown = reach
          pending = reach
          while pending != wp.uint32(0):
            tree = _lowest_set_bit(pending)
            pending &= ~(wp.uint32(1) << wp.uint32(tree))
            grown |= edge_sh[tree]
          if grown == reach:
            break
          reach = grown
        root = _lowest_set_bit(reach)
    is_root = int(0)
    if has_edge == 1 and root == tid:
      is_root = 1
    root_mask = _ballot(is_root)
    if tid < ntree:
      island_nv_out[worldid, tid] = 0
    if tid < ntree:
      if has_edge == 1:
        island = _popc(root_mask & ((wp.uint32(1) << wp.uint32(root)) - wp.uint32(1)))
        tree_island_out[worldid, tid] = island
        if is_root == 1:
          island_nv = int(0)
          pending = reach
          while pending != wp.uint32(0):
            tree = _lowest_set_bit(pending)
            pending &= ~(wp.uint32(1) << wp.uint32(tree))
            island_nv += tree_dofnum_sh[tree]
          island_nv_out[worldid, island] = island_nv
      else:
        tree_island_out[worldid, tid] = -1
    if tid == 0:
      nisland_out[worldid] = _popc(root_mask)

  return kernel


# resident CTAs per SM requested from the compiler for the integration kernel: 7 CTAs/SM hold 1024
# worlds in one wave on 170 SMs (<= 72 registers; the finalize variant's shared arenas stay under
# the ~13 KB that seven CTAs can share on a 100 KB SM)
_FORWARD_C_MIN_BLOCKS = 7


@cache_kernel
def _forward_c_kernel(NB: int, NV: int, NU: int, NT: int, NSMALL: int, NBIGDOF: int, NBIG: int, REFRESH: bool, PUBLISH: bool):
  """forward_c kernel factory; ``NU`` is the actuator capacity (a power of two >= nu)."""
  BLOCK = NV
  SMALL_SLOT = NSMALL * NSMALL + NSMALL
  BIG_SLOT = NBIGDOF * NBIGDOF + NBIGDOF
  FAC_SIZE = NT * SMALL_SLOT + NBIG * BIG_SLOT

  @wp.func_native(snippet="values.data(wp::tile_coord(index)) = value;")
  def _st_fac(values: wp.tile[float, FAC_SIZE], index: int, value: float): ...

  @wp.func_native(snippet="values.data(wp::tile_coord(index)) = value;")
  def _st_dof_f(values: wp.tile[float, NV], index: int, value: float): ...

  @wp.func_native(snippet="values.data(wp::tile_coord(index)) = value;")
  def _st_act_f(values: wp.tile[float, NU], index: int, value: float): ...

  @wp.func_native(snippet="values.data(wp::tile_coord(index)) = value;")
  def _st_act_i(values: wp.tile[int, NU], index: int, value: int): ...

  @wp.func_native(snippet="values.data(wp::tile_coord(index)) = value;")
  def _st_body_sv(values: wp.tile[wp.spatial_vector, NB], index: int, value: wp.spatial_vector): ...

  @wp.kernel(module="unique", enable_backward=False, grid_stride=False, launch_bounds=(BLOCK, _FORWARD_C_MIN_BLOCKS))
  def kernel(
    # Model:
    nbody: int,
    nv: int,
    nu: int,
    njnt: int,
    ntree: int,
    nvtree_max: int,
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
    act_vel_sh = wp.tile_empty(shape=(NU,), dtype=float, storage="shared")
    act_dof_sh = wp.tile_empty(shape=(NU,), dtype=int, storage="shared")
    tree_asleep_sh = wp.tile_empty(shape=(NT,), dtype=int, storage="shared")
    tree_island_sh = wp.tile_empty(shape=(NT,), dtype=int, storage="shared")
    tree_dofnum_sh = wp.tile_empty(shape=(NT,), dtype=int, storage="shared")
    tree_base_sh = wp.tile_empty(shape=(NT,), dtype=int, storage="shared")
    can_sleep_sh = wp.tile_empty(shape=(NT,), dtype=int, storage="shared")
    tree_implicit_sh = wp.tile_empty(shape=(NT,), dtype=int, storage="shared")
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
    if tid == 0:
      _st_misc16_i(misc_sh, 15, 0)
    if is_tree:
      t_n = tree_dofnum[tid]
      t_start = tree_dofadr[tid]
      _st_tree32_i(tree_asleep_sh, tid, tree_asleep_out[worldid, tid])
      _st_tree32_i(tree_island_sh, tid, tree_island_in[worldid, tid])
      _st_tree32_i(tree_dofnum_sh, tid, t_n)
      _st_tree32_i(can_sleep_sh, tid, 1)
      _st_tree32_i(tree_implicit_sh, tid, 0)
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
    b_mocap_root = int(-1)
    if is_body:
      b_tree = body_treeid[tid]
      b_mocap_root = body_mocapid[body_rootid[tid]]
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
    d_dense = is_dof and qLD_block_adr[tid] != Q_LD_BLOCK_COMPACT
    d_base = int(0)
    d_n = int(1)
    d_i = int(0)
    d_start = int(0)
    if d_dense:
      d_base = tree_base_sh[d_tree]
      d_n = tree_dofnum_sh[d_tree]
      d_start = tree_dofadr[d_tree]
      d_i = tid - d_start

    # ------------------------------------------------------------ C1: implicitfast factor and solve
    # qDeriv = M - h * (actuator + damping velocity derivatives) touches the diagonal only for joint
    # transmissions; factor/solve as smooth.factor_solve_i with rhs Ma (support.mul_m from the
    # solver). The integrated acceleration of a dof stays with its own thread.
    qacc_int = d_qacc
    if implicit_factor != 0:
      # A dense tree whose dofs all have a zero velocity derivative would factor M itself and
      # solve M^-1 (M qacc) = qacc, so it takes qacc directly (fp32 round-off apart). The trees
      # that do carry a derivative are factored per warp with __syncwarp whenever each of them lies
      # within one warp; a straddling tree falls back to the block-barrier factor.
      diag = float(0.0)
      rhs = float(0.0)
      rowadr = int(0)
      rownnz = int(0)
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
        if qderiv != 0.0 and d_tree >= 0:
          _st_tree32_i(tree_implicit_sh, d_tree, 1)
          if d_dense and (d_start >> 5) != ((d_start + d_n - 1) >> 5):
            _st_misc16_i(misc_sh, 15, 1)
      _sync()
      d_factor = False
      if d_dense:
        d_factor = tree_implicit_sh[d_tree] != 0
      if is_dof:
        if qLD_block_adr[tid] == Q_LD_BLOCK_COMPACT:
          inverse = 1.0 / diag
          qacc_int = inverse * rhs
        elif d_factor:
          # this dof's row of the dense block (zeroed first: branching trees leave structural zeros)
          for c in range(d_n):
            _st_fac(fac_sh, d_base + d_i * d_n + c, 0.0)
          for k in range(rownnz - 1):
            col = M_colind[rowadr + k] - d_start
            _st_fac(fac_sh, d_base + d_i * d_n + col, M_in[worldid, rowadr + k])
          _st_fac(fac_sh, d_base + d_i * d_n + d_i, diag)
          _st_fac(fac_sh, d_base + d_n * d_n + d_i, rhs)
      _sync()
      # one column per dof thread (smooth._small_cholesky_factorize_solve_block order); the path is
      # block-uniform since every thread reads the same flag after the barrier
      if misc_sh[15] == 0:
        _dense_factor_solve_warp(fac_sh, d_factor, d_base, d_n, d_i, nvtree_max)
      else:
        _dense_factor_solve(fac_sh, d_factor, d_base, d_n, d_i, nvtree_max)
      if d_factor:
        qacc_int = fac_sh[d_base + d_n * d_n + d_i]

    # --------------------------------------------------------- C2: advance velocity, position, time
    if is_dof:
      d_qvel = d_qvel + qacc_int * timestep
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
    if wp.static(REFRESH and PUBLISH):
      # fwd_velocity on the post-sleep qvel: cdof/cinert/subtree_com are the pre-integration values.
      # Same ancestor-walk structure as forward_a's P6/P8: every body stores its own joint term,
      # then sums the terms of its chain from shared memory (one barrier per pass instead of one per
      # tree level). Bodies read their own joint's cdof and their own cinert from global memory, so
      # the per-dof and per-body arenas of forward_a are not needed here (one wave at W1024).
      cvel_sh = wp.tile_empty(shape=(NB,), dtype=wp.spatial_vector, storage="shared")
      cacc_sh = wp.tile_empty(shape=(NB,), dtype=wp.spatial_vector, storage="shared")
      parent_sh = wp.tile_empty(shape=(NB,), dtype=int, storage="shared")
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
      cdof = wp.spatial_vector()
      if is_dof:
        cdof = cdof_in[worldid, tid]
      for actid in range(tid, nu, BLOCK):
        gear0 = actuator_gear[worldid % actuator_gear.shape[0], actid][0]
        actuator_velocity_out[worldid, actid] = gear0 * qvel_sh[act_dof_sh[actid]]
      if tid == 0:
        _st_body_sv(cvel_sh, 0, wp.spatial_vector())
        cacc0 = wp.spatial_vector()
        if gravity_enabled:
          cacc0 = wp.spatial_vector(wp.vec3(0.0), -gravity)
        _st_body_sv(cacc_sh, 0, cacc0)
      # each body's own velocity term v = sum_k cdof_k qvel_k over its joint; free bodies (children
      # of the world, whose cvel is zero) also finish their cdof_dot and acceleration term here
      v_own = wp.spatial_vector()
      a_own = wp.spatial_vector()
      cdof_own = wp.spatial_vector()
      if is_body and tid > 0 and b_jntnum == 1:
        if j_type == JointType.FREE:
          for k in range(3):
            v_own += cdof_in[worldid, b_dofadr + k] * qvel_sh[b_dofadr + k]
            cdof_dot_out[worldid, b_dofadr + k] = wp.spatial_vector()
          for k in range(3, 6):
            cdof_k = cdof_in[worldid, b_dofadr + k]
            cdof_dot = math.motion_cross(v_own, cdof_k)
            cdof_dot_out[worldid, b_dofadr + k] = cdof_dot
            a_own += cdof_dot * qvel_sh[b_dofadr + k]
          for k in range(3, 6):
            v_own += cdof_in[worldid, b_dofadr + k] * qvel_sh[b_dofadr + k]
        else:
          cdof_own = cdof_in[worldid, b_dofadr]
          v_own = cdof_own * qvel_sh[b_dofadr]
      if is_body and tid > 0:
        _st_body_sv(cvel_sh, tid, v_own)
      _sync()

      # subtree end from the depth-first numbering (the rne gather below); cvel_parent = sum of the
      # ancestors' terms (cvel_sh[0] is zero) read along the chain; cdof_dot of a hinge/slide dof
      # uses the parent velocity
      subtree_end = int(0)
      cvel = wp.spatial_vector()
      if is_body:
        subtree_end = tid + 1
        while subtree_end < nbody:
          if parent_sh[subtree_end] < tid:
            break
          subtree_end += 1
        if tid > 0:
          p = int(b_parent)
          while p != 0:
            cvel += cvel_sh[p]
            p = parent_sh[p]
          if b_jntnum == 1 and j_type != JointType.FREE:
            cdof_dot = math.motion_cross(cvel, cdof_own)
            cdof_dot_out[worldid, b_dofadr] = cdof_dot
            a_own = cdof_dot * qvel_sh[b_dofadr]
          cvel += v_own
      _sync()
      if is_body and tid > 0:
        _st_body_sv(cacc_sh, tid, a_own)
      _sync()
      # cacc = gravity root + the ancestors' terms + own term, same chain read
      cacc = wp.spatial_vector()
      if is_body:
        cacc = cacc_sh[0]
        if tid > 0:
          p = int(b_parent)
          while p != 0:
            cacc += cacc_sh[p]
            p = parent_sh[p]
          cacc += a_own
      _sync()
      if is_body:
        cvel_out[worldid, tid] = cvel
        cacc_out[worldid, tid] = cacc
      # own composite inertia for the rne body force below; the passive phase hides the load
      cinert_own = vec10()
      if is_body and tid > 0:
        cinert_own = cinert_in[worldid, tid]

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

      # rne: body forces (the cacc arena is free since the chain read), backward gather, qfrc_bias
      if is_body:
        frc = wp.spatial_vector()
        if tid > 0:
          frc = math.inert_vec(cinert_own, cacc)
          frc += math.motion_cross_force(cvel, math.inert_vec(cinert_own, cvel))
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
      b_tree,
      b_mocap_root,
      d_body,
      d_tree,
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


def subset_launch_dim(d: Data, world_ids: wp.array | None, count: int | wp.array | None) -> tuple[int, wp.array | None]:
  """Validate a world subset and return ``(grid worlds, device count or None)`` for its launch.

  ``count`` may be a host int (the grid spans exactly ``count`` slots), a one-element int32 device
  array (the grid spans ``world_ids.shape[0]`` slots and the kernel exits the slots at or beyond the
  count: graph-safe, no host sync) or None (``world_ids.shape[0]`` slots, all valid).
  """
  if world_ids is None:
    return d.nworld, None
  if not isinstance(world_ids, wp.array) or world_ids.dtype != wp.int32 or world_ids.ndim != 1:
    raise TypeError("world_ids must be a one-dimensional wp.array of int32 world indices.")
  if world_ids.device != d.qpos.device:
    raise ValueError(f"world_ids must be on {d.qpos.device}, got {world_ids.device}.")
  capacity = world_ids.shape[0]
  if capacity > d.nworld:
    raise ValueError(f"world_ids holds {capacity} entries, more than the {d.nworld} worlds of Data.")
  if count is None:
    return capacity, None
  if isinstance(count, wp.array):
    if count.dtype != wp.int32 or count.shape != (1,):
      raise TypeError("a device count must be a wp.array of shape (1,) and dtype int32.")
    if count.device != world_ids.device:
      raise ValueError(f"count must be on {world_ids.device}, got {count.device}.")
    return capacity, count
  count = int(count)
  if count < 0 or count > capacity:
    raise ValueError(f"count {count} is outside [0, {capacity}] (the world_ids capacity).")
  return count, None


@event_scope
def forward_a(
  m: Model,
  d: Data,
  groups: ContactGroups | None = None,
  world_ids: wp.array | None = None,
  count: int | wp.array | None = None,
  run_wake: bool | None = None,
):
  """Fused sleep wake/update, position, velocity and actuation stages (one CTA per world).

  Replaces ``sleep.wake`` + ``sleep.update_sleep_trees``, ``smooth.kinematics``, ``smooth.com_pos``,
  ``smooth.crb``, ``smooth.transmission``, ``fwd_velocity`` (actuator velocity, ``com_vel``, passive
  forces, ``rne``) and ``fwd_actuation`` for models accepted by :func:`fused_world`. The body/dof
  awake arrays are published once, by ``forward_m`` after ``wake_equality``. ``groups`` are the
  contact buckets shared with ``forward_m`` (their per-world counts are zeroed here). The wake pass
  is skipped when ``Option.run_sleep_wake`` is False (the caller already ran ``sleep.wake``);
  ``run_wake`` overrides that option (``tree_awake`` is republished from ``tree_asleep``
  regardless). :data:`DERIVED_FIELDS` are left stale when :func:`publish_derived` is False.

  With ``world_ids`` (int32 world indices) only those worlds run, CTA ``slot`` serving world
  ``world_ids[slot]``; ``count`` bounds the valid entries as a host int or a one-element int32
  device array (see :func:`subset_launch_dim`). Every other world is left untouched.
  """
  if groups is None:
    groups = contact_groups(d)
  if run_wake is None:
    run_wake = getattr(m.opt, "run_sleep_wake", True)
  publish = publish_derived(m)
  grid_worlds, world_count = subset_launch_dim(d, world_ids, count)
  wp.launch(
    _forward_a_kernel(NBODY_CAP, NV_CAP, _pow2_at_least(m.nu), NTREE_CAP, publish, world_ids is not None),
    dim=(grid_worlds, NV_CAP),
    inputs=[
      m.nbody,
      m.nv,
      m.nu,
      m.ntree,
      m.ngeom,
      m.nsite,
      int(m.ngravcomp > 0),
      int(run_wake),
      m.opt.disableflags,
      m.opt.gravity,
      m.qpos0,
      m.qpos_spring,
      m.fused_world_body_info,
      m.fused_world_dof_info,
      m.fused_world_act_info,
      m.body_pos,
      m.body_quat,
      m.body_ipos,
      m.body_iquat,
      m.body_mass,
      m.body_subtreemass,
      m.body_inertia,
      m.body_gravcomp,
      m.jnt_pos,
      m.jnt_axis,
      m.jnt_stiffness,
      m.jnt_stiffnesspoly,
      m.jnt_actfrcrange,
      m.dof_armature,
      m.dof_damping,
      m.dof_dampingpoly,
      m.tree_sleep_policy,
      m.geom_bodyid,
      m.geom_pos,
      m.geom_quat,
      m.site_bodyid,
      m.site_pos,
      m.site_quat,
      m.actuator_gear,
      m.actuator_gainprm,
      m.actuator_biasprm,
      m.actuator_ctrlrange,
      m.actuator_forcerange,
      d.qpos,
      d.qvel,
      d.qfrc_applied,
      d.xfrc_applied,
      d.mocap_pos,
      d.mocap_quat,
      d.ctrl,
      world_ids,
      world_count,
    ],
    outputs=[
      d.tree_asleep,
      d.tree_awake,
      groups.count,
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
def forward_b(m: Model, d: Data, *, compact_maps: bool = True):
  """Fused acceleration stage as one CTA per world.

  ``qfrc_smooth``, ``xfrc_accumulate``, the per-tree factor/solve with the sleeping-tree freeze and,
  with ``compact_maps``, the active-DOF compaction maps (``island.update_active_dofs``). The
  per-world solver derives its own slot map, so its path passes False and lets the stock fallback
  rebuild the maps on demand; the packed factor ``qLD`` is published only with
  :func:`publish_derived` (the stock fallback re-factors).
  """
  wp.launch(
    _forward_b_kernel(NBODY_CAP, NV_CAP, NTREE_CAP, NVTREE_SMALL, NVTREE_CAP, NBIG_CAP, publish_derived(m), bool(compact_maps)),
    dim=(d.nworld, NV_CAP),
    inputs=[
      m.nbody,
      m.nv,
      m.ntree,
      m.fused_world_nvtree_max,
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


@event_scope
def forward_m(m: Model, d: Data, groups: ContactGroups | None = None):
  """Fused constraint assembly and sleep/island bookkeeping between ``forward_a`` and ``forward_b``.

  Replaces ``constraint.make_constraint`` (JOINT equalities, dof friction, slide/hinge limits and
  pyramidal contact rows), ``sleep.wake_equality``, ``sleep.update_sleep`` and ``island.island`` for
  models accepted by :func:`fused_world`; one bucket launch sorts the contacts by world first. Pass
  the ``groups`` given to ``forward_a`` (which zeroes the counts) or let this function allocate
  them.
  """
  if groups is None:
    groups = contact_groups(d)
    groups.count.zero_()
  _bucket_contacts(m, d, groups)
  _launch_forward_m(m, d, groups)


def _launch_forward_m(m: Model, d: Data, groups: ContactGroups):
  contact_frame_2d = wp.array(
    ptr=d.contact.frame.ptr, dtype=wp.vec3, shape=(d.naconmax, 3), device=d.contact.frame.device, copy=False
  )
  wp.launch(
    _forward_m_kernel(NBODY_CAP, NV_CAP, NTREE_CAP, publish_derived(m)),
    dim=(d.nworld, NV_CAP),
    inputs=[
      m.nv,
      m.nbody,
      m.ntree,
      m.neq,
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
      d.contact.pos,
      contact_frame_2d,
      d.contact.friction,
      d.contact.solref,
      d.contact.solimp,
      groups.capacity,
      groups.count,
      groups.ids,
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
    _forward_c_kernel(
      NBODY_CAP, NV_CAP, _pow2_at_least(m.nu), NTREE_CAP, NVTREE_SMALL, NVTREE_CAP, NBIG_CAP, bool(finalize), publish_derived(m)
    ),
    dim=(d.nworld, NV_CAP),
    inputs=[
      m.nbody,
      m.nv,
      m.nu,
      m.njnt,
      m.ntree,
      m.fused_world_nvtree_max,
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
