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

"""Per-world component-local constraint solver (one CTA per world).

Replaces the compact Newton recurrence on the sleeping path with a single launch:
one 128-thread CTA per world derives the constraint components from
``tree_awake``/``tree_island`` (the same active predicate as
``island._compact_dof_layout``), projects the sparse ``efc.J`` onto the active DOFs
exactly like the compact kernels (``if colind < 0: continue``) and solves the
pyramidal piecewise-quadratic problem per component with semismooth Newton steps
on the active set (``q -= H^-1 grad`` with ``H = M + J^T D_active J``), which is the
active-set fixed point written as a residual correction.

Every iterate is certified with the history-free clauses of ``solver._solve_done``
(rescaled gradient or half Newton decrement below ``d.ctol`` with
``_rescale(nvmax_pad, meaninertia)``), so a certified world stops at a point at which
the stock solver would also have stopped. Uncertified worlds and worlds that exceed
the compile-time capacities (component DOFs > 32, active DOFs > 128, ``nf > 0``) are
flagged in ``stock_world`` and counted in ``nstock`` for the stock fallback.

All reductions run in a fixed order (stable row bucketing, thread-per-entry Hessian
accumulation, warp-0 shuffles), so outputs are replay-stable for identical inputs.
"""

from __future__ import annotations

import dataclasses

import warp as wp

from mujoco_warp._src import types

BLOCK_DIM = 256
TILE_ROWS = 128
COMP_DOF_CAP = 32
NCDOF_CAP = 128
NTREE_CAP = 32
ROUND_CAP = 10
PACKED_TOTAL_CAP = 640

STATUS_CERTIFIED = 0
STATUS_UNCERTIFIED = 1
STATUS_NOT_PD = 3
STATUS_NONFINITE = 4
# capacity routing sub-codes (>= 20): nisland, friction rows, component DOFs, active DOFs,
# packed Hessian total, row spanning islands
STATUS_CAPACITY = 20

_NATIVE_TEMPLATE = r"""
{
  constexpr int BLOCK = __BLOCK__;
  constexpr int TILE = __TILE__;
  constexpr int CDOF = __CDOF__;
  constexpr int NCDOF = __NCDOF__;
  constexpr int NTREE = __NTREE__;
  constexpr int NJMAX = __NJMAX__;
  constexpr int NV = __NV__;
  constexpr int ROUND_CAP = __ROUND_CAP__;
  constexpr int PACKED_TOTAL = __PACKED_TOTAL__;
  constexpr int DEBUG_EXIT = __DEBUG_EXIT__;
  constexpr int NNZ_HALF = 8;  // per-thread prefetch depth (two threads per staged row)
  constexpr int NNZ_PREFETCH = 2 * NNZ_HALF;
  constexpr int STATE_SATISFIED = __STATE_SATISFIED__;
  constexpr int STATE_QUADRATIC = __STATE_QUADRATIC__;
  constexpr int ROW_NONE = 255;
  constexpr int STATUS_CERTIFIED = 0, STATUS_UNCERTIFIED = 1, STATUS_NOT_PD = 3, STATUS_NONFINITE = 4;
  constexpr int STATUS_CAP_NISLAND = 20, STATUS_CAP_FRICTION = 21, STATUS_CAP_COMP_DOF = 22, STATUS_CAP_NCDOF = 23,
                STATUS_CAP_PACKED = 24, STATUS_CAP_ROW_ISLANDS = 25, STATUS_CAP_ROW_NNZ = 26;
  constexpr int NWARPS = BLOCK / 32;
  static_assert(BLOCK % 32 == 0, "block must be whole warps");
  static_assert(NTREE <= 32, "island bitsets are 32 wide");
  static_assert(NJMAX + NWARPS * NTREE * 4 + 16 <= TILE * CDOF * 4, "bucketing scratch must fit in the tile arena");
  const unsigned FULL = 0xffffffffu;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const unsigned lanemask_lt = (1u << lane) - 1u;

  __shared__ __align__(16) float sh_tile_J[TILE * CDOF];
  __shared__ float sh_tile_F[TILE];
  __shared__ float sh_H[PACKED_TOTAL];
  __shared__ unsigned short sh_Melem[PACKED_TOTAL];  // M CSR address per packed entry (0xffff: zero)
  __shared__ unsigned short sh_entry_ab[PACKED_TOTAL];  // (a << 8) | b
  __shared__ unsigned char sh_entry_comp[PACKED_TOTAL];
  __shared__ float sh_q[NCDOF];
  __shared__ float sh_qfrc[NCDOF];
  __shared__ unsigned char sh_slot_comp[NCDOF];
  __shared__ unsigned short sh_slot_dof[NCDOF];
  __shared__ unsigned short sh_row_order[NJMAX];
  __shared__ short sh_dof_slot[NV];
  __shared__ signed char sh_tree_comp[NTREE];
  __shared__ unsigned char sh_comp_ndof[NTREE];
  __shared__ unsigned char sh_comp_begin[NTREE];
  __shared__ short sh_comp_pbegin[NTREE + 1];
  __shared__ int sh_comp_row_begin[NTREE + 1];
  __shared__ unsigned short sh_comp_nrow[NTREE];
  __shared__ int sh_bucket_count[NTREE];
  __shared__ int sh_bucket_fill[NTREE];
  __shared__ unsigned char sh_comp_done[NTREE];
  __shared__ unsigned char sh_comp_status[NTREE];
  __shared__ unsigned char sh_comp_rounds[NTREE];
  __shared__ float sh_comp_gd[NTREE];
  __shared__ float sh_comp_dec[NTREE];
  __shared__ int sh_flags[8];
  // per-slot gradient / Newton step alias the tile arena: they are live only between the last
  // tile of a round and the next round's staging
  float* const sh_grad = sh_tile_J;
  float* const sh_s = sh_tile_J + NCDOF;
  static_assert(2 * NCDOF <= TILE * CDOF, "grad/s alias must fit in the tile arena");
  // bucketing scratch aliases the tile arena (dead before any tile is staged)
  unsigned char* const sh_row_comp = reinterpret_cast<unsigned char*>(sh_tile_J);
  int* const sh_warp_cnt = reinterpret_cast<int*>(sh_tile_J + ((NJMAX + 15) / 16) * 4);

  // ---- raw pointers and element strides (all 4-byte element types) ----
  const int* const ne_p = reinterpret_cast<const int*>(ne_in.data);
  const int* const nf_p = reinterpret_cast<const int*>(nf_in.data);
  const int* const nefc_p = reinterpret_cast<const int*>(nefc_in.data);
  const int* const nisland_p = reinterpret_cast<const int*>(nisland_in.data);
  const int* const tree_awake_p = reinterpret_cast<const int*>(tree_awake_in.data);
  const int s_tree_awake = tree_awake_in.strides[0] / 4;
  const int* const tree_island_p = reinterpret_cast<const int*>(tree_island_in.data);
  const int s_tree_island = tree_island_in.strides[0] / 4;
  const int* const dof_treeid_p = reinterpret_cast<const int*>(dof_treeid.data);
  const int* const tree_dofadr_p = reinterpret_cast<const int*>(tree_dofadr.data);
  const int* const tree_dofnum_p = reinterpret_cast<const int*>(tree_dofnum.data);
  const int* const M_elemid_p = reinterpret_cast<const int*>(M_elemid.data);
  const int s_M_elemid = M_elemid.strides[0] / 4;
  const int* const mulm_rowadr_p = reinterpret_cast<const int*>(M_mulm_rowadr.data);
  const int* const mulm_col_p = reinterpret_cast<const int*>(M_mulm_col.data);
  const int* const mulm_madr_p = reinterpret_cast<const int*>(M_mulm_madr.data);
  const float* const M_p = reinterpret_cast<const float*>(M_in.data) + worldid * (M_in.strides[0] / 4);
  const int* const rownnz_p = reinterpret_cast<const int*>(efc_J_rownnz_in.data) + worldid * (efc_J_rownnz_in.strides[0] / 4);
  const int* const rowadr_p = reinterpret_cast<const int*>(efc_J_rowadr_in.data) + worldid * (efc_J_rowadr_in.strides[0] / 4);
  const int* const colind_p = reinterpret_cast<const int*>(efc_J_colind_in.data) + worldid * (efc_J_colind_in.strides[0] / 4);
  const float* const J_p = reinterpret_cast<const float*>(efc_J_in.data) + worldid * (efc_J_in.strides[0] / 4);
  const float* const D_p = reinterpret_cast<const float*>(efc_D_in.data) + worldid * (efc_D_in.strides[0] / 4);
  const float* const aref_p = reinterpret_cast<const float*>(efc_aref_in.data) + worldid * (efc_aref_in.strides[0] / 4);
  const float* const qacc_smooth_p = reinterpret_cast<const float*>(qacc_smooth_in.data) + worldid * (qacc_smooth_in.strides[0] / 4);
  const float* const qfrc_smooth_p = reinterpret_cast<const float*>(qfrc_smooth_in.data) + worldid * (qfrc_smooth_in.strides[0] / 4);
  const float* const warm_p = reinterpret_cast<const float*>(qacc_warmstart_in.data) + worldid * (qacc_warmstart_in.strides[0] / 4);
  const float* const meaninertia_p = reinterpret_cast<const float*>(stat_meaninertia.data);
  const float* const ctol_p = reinterpret_cast<const float*>(ctol_in.data);
  float* const qacc_o = reinterpret_cast<float*>(qacc_out.data) + worldid * (qacc_out.strides[0] / 4);
  float* const qfrc_o = reinterpret_cast<float*>(qfrc_constraint_out.data) + worldid * (qfrc_constraint_out.strides[0] / 4);
  float* const force_o = reinterpret_cast<float*>(efc_force_out.data) + worldid * (efc_force_out.strides[0] / 4);
  int* const state_o = reinterpret_cast<int*>(efc_state_out.data) + worldid * (efc_state_out.strides[0] / 4);
  float* const Ma_o = reinterpret_cast<float*>(efc_Ma_out.data) + worldid * (efc_Ma_out.strides[0] / 4);
  int* const niter_o = reinterpret_cast<int*>(solver_niter_out.data);
  int* const stock_o = reinterpret_cast<int*>(stock_world_out.data);
  int* const nstock_o = reinterpret_cast<int*>(nstock_out.data);
  int* const status_o = reinterpret_cast<int*>(world_status_out.data);
  float* const gradient_o = reinterpret_cast<float*>(world_gradient_out.data);
  float* const decrement_o = reinterpret_cast<float*>(world_decrement_out.data);

  // stock clamps every row loop to njmax (solver.py: wp.min(nefc, njmax))
  const int nefc = min(nefc_p[worldid], njmax);
  const int ne = ne_p[worldid];
  const int nf = nf_p[worldid];
  const int nisland = nisland_p[worldid];
  const float meaninertia = meaninertia_p[worldid % (int)stat_meaninertia.shape[0]];
  const float ctol = ctol_p[0];
  const float scale = meaninertia * (float)nv_scale;

  // ---------------------------------------------------------------- phase 0: components
  if (tid < 8) sh_flags[tid] = 0;
  for (int g = tid; g < NV; g += BLOCK) sh_dof_slot[g] = -1;
  if (tid < NTREE) {
    sh_comp_ndof[tid] = 0;
    sh_comp_begin[tid] = 0;
    sh_comp_pbegin[tid] = 0;
    sh_bucket_count[tid] = 0;
    sh_bucket_fill[tid] = 0;
    sh_comp_done[tid] = 1;
    sh_comp_status[tid] = STATUS_CERTIFIED;
    sh_comp_rounds[tid] = 0;
    sh_comp_gd[tid] = 0.0f;
    sh_comp_dec[tid] = 0.0f;
    signed char comp = -1;
    if (tid < ntree) {
      const int awake = tree_awake_p[worldid * s_tree_awake + tid];
      const int isl = tree_island_p[worldid * s_tree_island + tid];
      // ACTIVE predicate of island._compact_dof_layout / _map_compact_dofs
      if (awake == 1 && isl >= 0) comp = (signed char)isl;
    }
    sh_tree_comp[tid] = comp;
  }
  __syncthreads();
  // per island: active DOF count (thread per island)
  if (tid < NTREE) {
    int nd = 0;
    if (tid < nisland) {
      for (int t2 = 0; t2 < ntree; ++t2) {
        if (sh_tree_comp[t2] == tid) nd += tree_dofnum_p[t2];
      }
    }
    sh_bucket_count[tid] = nd;  // scratch: island DOF count
  }
  __syncthreads();
  if (tid == 0) {
    int status = STATUS_CERTIFIED;
    if (nisland > NTREE) status = STATUS_CAP_NISLAND;
    if (nf > 0) status = STATUS_CAP_FRICTION;
    int ncdof = 0;
    int npacked = 0;
    int ncomp_active = 0;
    const int nisl = min(nisland, NTREE);
    for (int isl = 0; isl < nisl; ++isl) {
      const int nd = sh_bucket_count[isl];
      sh_comp_begin[isl] = (unsigned char)min(ncdof, 255);
      sh_comp_ndof[isl] = (unsigned char)min(nd, 255);
      sh_comp_pbegin[isl] = (short)min(npacked, 32767);
      if (nd > CDOF) status = STATUS_CAP_COMP_DOF;
      if (nd > 0) {
        ++ncomp_active;
        sh_comp_done[isl] = 0;
      }
      ncdof += nd;
      npacked += nd * (nd + 1) / 2;
    }
    sh_comp_pbegin[nisl] = (short)min(npacked, 32767);
    if (ncdof > NCDOF) status = STATUS_CAP_NCDOF;
    if (npacked > PACKED_TOTAL) status = STATUS_CAP_PACKED;
    sh_flags[0] = status;
    sh_flags[1] = ncdof;
    sh_flags[2] = ncomp_active;
    sh_flags[3] = npacked;
  }
  __syncthreads();
  // per tree: slot assignment (trees of an island in increasing tree order)
  if (tid < ntree && sh_flags[0] == STATUS_CERTIFIED) {
    const int isl = sh_tree_comp[tid];
    if (isl >= 0) {
      int offset = 0;
      for (int t2 = 0; t2 < tid; ++t2) {
        if (sh_tree_comp[t2] == isl) offset += tree_dofnum_p[t2];
      }
      const int base = (int)sh_comp_begin[isl] + offset;
      const int adr = tree_dofadr_p[tid];
      const int num = tree_dofnum_p[tid];
      for (int j = 0; j < num; ++j) {
        const int slot = base + j;
        if (slot < NCDOF) {
          sh_dof_slot[adr + j] = (short)slot;
          sh_slot_dof[slot] = (unsigned short)adr + (unsigned short)j;
          sh_slot_comp[slot] = (unsigned char)isl;
        }
      }
    }
  }
  if (tid < NTREE) sh_bucket_count[tid] = 0;
  __syncthreads();

  // row -> component (island of its first active column); all active columns must agree
  if (sh_flags[0] == STATUS_CERTIFIED) {
    for (int r = tid; r < nefc; r += BLOCK) {
      const int nnz = rownnz_p[r];
      const int adr = rowadr_p[r];
      if (nnz > NNZ_PREFETCH) sh_flags[0] = STATUS_CAP_ROW_NNZ;
      int cols[NNZ_PREFETCH];
#pragma unroll
      for (int k = 0; k < NNZ_PREFETCH; ++k) cols[k] = (k < nnz) ? colind_p[adr + k] : 0;
      int comp = ROW_NONE;
      bool bad = false;
#pragma unroll
      for (int k = 0; k < NNZ_PREFETCH; ++k) {
        if (k < nnz && sh_dof_slot[cols[k]] >= 0) {
          const int isl = (int)sh_tree_comp[dof_treeid_p[cols[k]]];
          if (comp == ROW_NONE) comp = isl;
          else if (isl != comp) bad = true;
        }
      }
      sh_row_comp[r] = (unsigned char)comp;
      if (comp != ROW_NONE) atomicAdd(&sh_bucket_count[comp], 1);
      if (bad) sh_flags[0] = STATUS_CAP_ROW_ISLANDS;
    }
  }
  __syncthreads();
  if (sh_flags[0] != STATUS_CERTIFIED) {
    if (tid == 0) {
      stock_o[worldid] = 1;
      atomicAdd(nstock_o, 1);
      status_o[worldid] = sh_flags[0];
      gradient_o[worldid] = 0.0f;
      decrement_o[worldid] = 0.0f;
    }
    return;
  }
  const int ncdof = sh_flags[1];
  const int ncomp_active = sh_flags[2];
  const int npacked = sh_flags[3];
  if (tid == 0) {
    int acc = 0;
    for (int isl = 0; isl < nisland; ++isl) {
      sh_comp_row_begin[isl] = acc;
      acc += sh_bucket_count[isl];
    }
    sh_comp_row_begin[nisland] = acc;
  }
  // packed-entry -> component table, M_c blocks, q0
  for (int E = tid; E < npacked; E += BLOCK) {
    int c = 0;
    while (c + 1 < nisland && (int)sh_comp_pbegin[c + 1] <= E) ++c;
    const int e = E - sh_comp_pbegin[c];
    int a = (int)((sqrtf(8.0f * (float)e + 1.0f) - 1.0f) * 0.5f);
    while ((a + 1) * (a + 2) / 2 <= e) ++a;
    while (a * (a + 1) / 2 > e) --a;
    const int b = e - a * (a + 1) / 2;
    sh_entry_comp[E] = (unsigned char)c;
    sh_entry_ab[E] = (unsigned short)((a << 8) | b);
    const int sb = sh_comp_begin[c];
    const int ga = sh_slot_dof[sb + a];
    const int gb = sh_slot_dof[sb + b];
    const int elemid = M_elemid_p[max(ga, gb) * s_M_elemid + min(ga, gb)];
    sh_Melem[E] = elemid >= 0 ? (unsigned short)elemid : (unsigned short)0xffff;
  }
  for (int s = tid; s < ncdof; s += BLOCK) {
    const int g = sh_slot_dof[s];
    sh_q[s] = warmstart ? warm_p[g] : qacc_smooth_p[g];
  }
  __syncthreads();
  // stable bucketing: rows keep their efc order inside each component
  for (int base = 0; base < nefc; base += BLOCK) {
    for (int i = tid; i < NWARPS * NTREE; i += BLOCK) sh_warp_cnt[i] = 0;
    __syncthreads();
    const int r = base + tid;
    const int comp = (r < nefc) ? (int)sh_row_comp[r] : ROW_NONE;
    const unsigned peers = __match_any_sync(FULL, comp);
    const int rank = __popc(peers & lanemask_lt);
    if (comp != ROW_NONE && rank == 0) sh_warp_cnt[warp * NTREE + comp] = __popc(peers);
    __syncthreads();
    if (comp != ROW_NONE) {
      int off = sh_comp_row_begin[comp] + sh_bucket_fill[comp] + rank;
      for (int w2 = 0; w2 < warp; ++w2) off += sh_warp_cnt[w2 * NTREE + comp];
      sh_row_order[off] = (unsigned short)r;
    }
    __syncthreads();
    if (tid < NTREE) {
      int tot = 0;
      for (int w2 = 0; w2 < NWARPS; ++w2) tot += sh_warp_cnt[w2 * NTREE + tid];
      sh_bucket_fill[tid] += tot;
    }
    __syncthreads();
  }

  if (DEBUG_EXIT == 1) { if (tid == 0) status_o[worldid] = 0; return; }
  // ---------------------------------------------------------------- phase 1: all components per round
  const float comp_tol = ctol / (float)max(ncomp_active, 1);
  if (tid < NTREE) {
    sh_comp_nrow[tid] = (tid < nisland) ? (unsigned short)(sh_comp_row_begin[tid + 1] - sh_comp_row_begin[tid]) : 0;
  }
  if (tid == 0) {
    sh_flags[4] = sh_comp_row_begin[nisland];
    sh_flags[5] = 0;
  }
  __syncthreads();
  int nrow_open = sh_flags[4];
  const int t = tid >> 1;
  const int half = tid & 1;
  // registers holding the prefetched row of the tile about to be staged (two threads per row)
  int pf_r = 0;
  int pf_nnz = 0;
  float pf_D = 0.0f;
  float pf_aref = 0.0f;
  int pf_cols[NNZ_HALF];
  float pf_vals[NNZ_HALF];
  auto prefetch_tile = [&](const int tb_) {
    const int rows_ = min(TILE, nrow_open - tb_);
    pf_r = 0;
    pf_nnz = 0;
    pf_D = 0.0f;
    pf_aref = 0.0f;
#pragma unroll
    for (int i = 0; i < NNZ_HALF; ++i) {
      pf_cols[i] = 0;
      pf_vals[i] = 0.0f;
    }
    if (t < rows_) {
      pf_r = sh_row_order[tb_ + t];
      pf_nnz = rownnz_p[pf_r];
      const int adr = rowadr_p[pf_r];
      pf_D = D_p[pf_r];
      pf_aref = aref_p[pf_r];
      const int k0 = half ? (pf_nnz + 1) / 2 : 0;
      const int k1 = half ? pf_nnz : (pf_nnz + 1) / 2;
#pragma unroll
      for (int i = 0; i < NNZ_HALF; ++i) {
        const int k = k0 + i;
        pf_cols[i] = (k < k1) ? colind_p[adr + k] : 0;
        pf_vals[i] = (k < k1) ? J_p[adr + k] : 0.0f;
      }
    }
  };

  for (int round = 0; round < ROUND_CAP; ++round) {
    // drop the rows of components that finished last round (stable, in place)
    if (sh_flags[5]) {
      if (tid == 0) {
        int acc = 0;
        for (int c = 0; c < nisland; ++c) {
          sh_bucket_count[c] = acc;  // new row begin
          if (!sh_comp_done[c]) acc += sh_comp_nrow[c];
        }
        sh_flags[4] = acc;
      }
      __syncthreads();
      for (int base = 0; base < nrow_open; base += BLOCK) {
        const int i = base + tid;
        int val = 0;
        int dst = -1;
        if (i < nrow_open) {
          int c = 0;
          while (c + 1 < nisland && i >= sh_comp_row_begin[c + 1]) ++c;
          if (!sh_comp_done[c]) {
            val = sh_row_order[i];
            dst = sh_bucket_count[c] + (i - sh_comp_row_begin[c]);
          }
        }
        __syncthreads();
        if (dst >= 0) sh_row_order[dst] = (unsigned short)val;
        __syncthreads();
      }
      if (tid < nisland) sh_comp_row_begin[tid] = sh_bucket_count[tid];
      if (tid == 0) {
        sh_comp_row_begin[nisland] = sh_flags[4];
        sh_flags[5] = 0;
      }
      __syncthreads();
      nrow_open = sh_flags[4];
    }
    for (int E = tid; E < npacked; E += BLOCK) {
      const unsigned short me = sh_Melem[E];
      sh_H[E] = (me != 0xffff) ? M_p[me] : 0.0f;
    }
    for (int s = tid; s < ncdof; s += BLOCK) {
      if (!sh_comp_done[sh_slot_comp[s]]) sh_qfrc[s] = 0.0f;
    }
    prefetch_tile(0);
    __syncthreads();

    // stage rows in tiles (two threads per row): classify at q, dense component-local J scaled
    // by sqrt(D) (0 for SATISFIED rows) and force / sqrt(D), so H += Jt Jt^T and J^T f = sum Ft Jt
    for (int tb = 0; tb < nrow_open; tb += TILE) {
      const int rows = min(TILE, nrow_open - tb);
      const bool own = t < rows;
      float* const Jt = sh_tile_J + t * CDOF;
      const int r = pf_r;
      const int k0 = half ? (pf_nnz + 1) / 2 : 0;
      const int k1 = half ? pf_nnz : (pf_nnz + 1) / 2;
      int slots[NNZ_HALF];
      float xh = 0.0f;
      int comp_half = ROW_NONE;
#pragma unroll
      for (int i = 0; i < NNZ_HALF; ++i) {
        const int s = (own && k0 + i < k1) ? (int)sh_dof_slot[pf_cols[i]] : -1;
        slots[i] = s;
        if (s >= 0) {
          comp_half = min(comp_half, (int)sh_slot_comp[s]);
          xh += pf_vals[i] * sh_q[s];
        }
      }
      // pair reduction (both halves of a row are adjacent lanes; executed by the whole warp)
      const int comp = min(comp_half, __shfl_xor_sync(FULL, comp_half, 1));
      const float x = xh + __shfl_xor_sync(FULL, xh, 1) - pf_aref;
      int sb = 0;
      float sd = 0.0f;
      if (own) {
        // _eval_constraint: equality -> QUADRATIC; limit/contact -> QUADRATIC iff jaref < 0
        const bool quad = (r < ne) || (x < 0.0f);
        sd = quad ? sqrtf(pf_D) : 0.0f;
        if (comp != ROW_NONE) {
          sb = sh_comp_begin[comp];
          const int n = sh_comp_ndof[comp];
          for (int a = half; a < n; a += 2) Jt[a] = 0.0f;
        }
        if (half == 0) sh_tile_F[t] = (quad && sd > 0.0f) ? (-pf_D * x) / sd : 0.0f;
      }
      __syncwarp();
      if (own && comp != ROW_NONE) {
#pragma unroll
        for (int i = 0; i < NNZ_HALF; ++i) {
          if (slots[i] >= 0) Jt[slots[i] - sb] = pf_vals[i] * sd;
        }
      }
      __syncthreads();
      // issue the next tile's global loads before this tile's accumulation
      if (tb + TILE < nrow_open) prefetch_tile(tb + TILE);
      if (DEBUG_EXIT != 5) {
        // H += Jt^T Jt: a lane pair per packed entry, each lane summing half of the component's
        // tile rows; the pair sum is a fixed two-term addition so the result is replay-stable
        const int npairs = ((2 * npacked + 31) / 32) * 32;
        for (int E2 = tid; E2 < npairs; E2 += BLOCK) {
          const int E = E2 >> 1;
          const int hh = E2 & 1;
          float acc = 0.0f;
          bool write = false;
          if (E < npacked) {
            const int c = sh_entry_comp[E];
            if (!sh_comp_done[c]) {
              const int rb = sh_comp_row_begin[c];
              const int t0 = max(0, rb - tb);
              const int t1 = min(rows, rb + (int)sh_comp_nrow[c] - tb);
              if (t1 > t0) {
                const int tm = (t0 + t1) >> 1;
                const int ta = hh ? tm : t0;
                const int tz = hh ? t1 : tm;
                const int ab = sh_entry_ab[E];
                const int a = ab >> 8;
                const int b = ab & 0xff;
#pragma unroll 4
                for (int tt = ta; tt < tz; ++tt) acc += sh_tile_J[tt * CDOF + a] * sh_tile_J[tt * CDOF + b];
                write = (hh == 0);
              }
            }
          }
          acc += __shfl_xor_sync(FULL, acc, 1);
          if (write) sh_H[E] += acc;
        }
        // J^T f += Jt^T Ft: a lane pair per active slot
        const int spairs = ((2 * ncdof + 31) / 32) * 32;
        for (int S2 = tid; S2 < spairs; S2 += BLOCK) {
          const int s = S2 >> 1;
          const int hh = S2 & 1;
          float acc = 0.0f;
          bool write = false;
          if (s < ncdof) {
            const int c = sh_slot_comp[s];
            if (!sh_comp_done[c]) {
              const int rb = sh_comp_row_begin[c];
              const int t0 = max(0, rb - tb);
              const int t1 = min(rows, rb + (int)sh_comp_nrow[c] - tb);
              if (t1 > t0) {
                const int tm = (t0 + t1) >> 1;
                const int ta = hh ? tm : t0;
                const int tz = hh ? t1 : tm;
                const int a = s - sh_comp_begin[c];
#pragma unroll 4
                for (int tt = ta; tt < tz; ++tt) acc += sh_tile_F[tt] * sh_tile_J[tt * CDOF + a];
                write = (hh == 0);
              }
            }
          }
          acc += __shfl_xor_sync(FULL, acc, 1);
          if (write) sh_qfrc[s] += acc;
        }
      }
      __syncthreads();
    }

    // grad = Ma - qfrc_smooth - J^T force per slot
    for (int s = tid; s < ncdof; s += BLOCK) {
      const int c = sh_slot_comp[s];
      if (sh_comp_done[c]) continue;
      const int sb = sh_comp_begin[c];
      const int n = sh_comp_ndof[c];
      const int pb = sh_comp_pbegin[c];
      const int a = s - sb;
      float ma = 0.0f;
      for (int b = 0; b < n; ++b) {
        const int hi = max(a, b);
        const int lo = min(a, b);
        const unsigned short me = sh_Melem[pb + hi * (hi + 1) / 2 + lo];
        if (me != 0xffff) ma += M_p[me] * sh_q[sb + b];
      }
      sh_grad[s] = ma - qfrc_smooth_p[sh_slot_dof[s]] - sh_qfrc[s];
    }
    __syncthreads();

    // per component: factor H, s = H^-1 grad, certificate, Newton step.
    // Small components (n <= SMALL_N) run fully serial on one lane of the last warp (no warp
    // syncs, up to 32 components at once); larger ones take a whole warp on the other warps.
    constexpr int SMALL_N = 9;
    if (DEBUG_EXIT == 3 || DEBUG_EXIT == 5) {
      if (tid < nisland) {
        sh_comp_done[tid] = 1;
        sh_comp_rounds[tid] = 1;
        sh_flags[5] = 1;
      }
    } else if (warp == NWARPS - 1) {
      const int c = lane;
      if (c < nisland && !sh_comp_done[c] && sh_comp_ndof[c] <= SMALL_N) {
        const int n = sh_comp_ndof[c];
        const int sb = sh_comp_begin[c];
        float* const Hc = sh_H + sh_comp_pbegin[c];
        float* const gc = sh_grad + sb;
        float* const sc = sh_s + sb;
        bool ok = true;
        for (int p = 0; p < n; ++p) {
          const int rp = p * (p + 1) / 2;
          float d = Hc[rp + p];
          for (int k = 0; k < p; ++k) {
            const float l = Hc[rp + k];
            d -= l * l;
          }
          if (!(d > 0.0f) || !isfinite(d)) {
            ok = false;
            break;
          }
          const float ld = sqrtf(d);
          const float inv = 1.0f / ld;
          Hc[rp + p] = ld;
          for (int i = p + 1; i < n; ++i) {
            const int ri = i * (i + 1) / 2;
            float v = Hc[ri + p];
            for (int k = 0; k < p; ++k) v -= Hc[ri + k] * Hc[rp + k];
            Hc[ri + p] = v * inv;
          }
        }
        if (!ok) {
          sh_comp_status[c] = STATUS_NOT_PD;
          sh_comp_done[c] = 1;
          sh_comp_rounds[c] = round + 1;
          sh_flags[5] = 1;
        } else {
          float gd = 0.0f;
          for (int i = 0; i < n; ++i) {
            const int ri = i * (i + 1) / 2;
            const float g = gc[i];
            gd += g * g;
            float y = g;
            for (int k = 0; k < i; ++k) y -= Hc[ri + k] * sc[k];
            sc[i] = y / Hc[ri + i];
          }
          for (int i = n - 1; i >= 0; --i) {
            float y = sc[i];
            for (int k = i + 1; k < n; ++k) y -= Hc[k * (k + 1) / 2 + i] * sc[k];
            sc[i] = y / Hc[i * (i + 1) / 2 + i];
          }
          float sv = 0.0f;
          for (int i = 0; i < n; ++i) sv += sc[i] * gc[i];
          const bool finite_ok = isfinite(gd) && isfinite(sv);
          const float grad_r = sqrtf(gd) / scale;
          const float mi_r = 0.5f * sv / scale;
          const bool pass = finite_ok && ((grad_r < comp_tol) || (mi_r < comp_tol));
          const bool last = (round + 1 >= ROUND_CAP);
          sh_comp_gd[c] = gd;
          sh_comp_dec[c] = sv;
          sh_comp_rounds[c] = round + 1;
          if (!finite_ok) {
            sh_comp_status[c] = STATUS_NONFINITE;
            sh_comp_done[c] = 1;
            sh_flags[5] = 1;
          } else if (pass || last) {
            sh_comp_done[c] = 1;
            sh_flags[5] = 1;
          } else {
            // semismooth Newton step on the current active set
            for (int i = 0; i < n; ++i) sh_q[sb + i] -= sc[i];
          }
        }
      }
    } else {
      constexpr int NBIG_WARPS = (NWARPS > 1) ? (NWARPS - 1) : 1;
      for (int c = warp; c < nisland; c += NBIG_WARPS) {
        if (sh_comp_done[c] || sh_comp_ndof[c] <= SMALL_N) continue;
        const int n = sh_comp_ndof[c];
        const int sb = sh_comp_begin[c];
        float* const Hc = sh_H + sh_comp_pbegin[c];
        float* const gc = sh_grad + sb;
        float* const sc = sh_s + sb;
        int ok = 1;
        for (int p = 0; p < n; ++p) {
          const int dp = p * (p + 1) / 2 + p;
          const float d = Hc[dp];
          if (!(d > 0.0f) || !isfinite(d)) {
            ok = 0;
            break;
          }
          const float ld = sqrtf(d);
          const float inv = 1.0f / ld;
          __syncwarp();
          if (lane == 0) Hc[dp] = ld;
          for (int i = p + 1 + lane; i < n; i += 32) Hc[i * (i + 1) / 2 + p] *= inv;
          __syncwarp();
          // trailing update, row by row: lane j updates H[i][p+1+j] for j <= i-p-1
          const int lj = p + 1 + lane;
          const float lpj = (lj < n) ? Hc[lj * (lj + 1) / 2 + p] : 0.0f;
          for (int i = p + 1; i < n; ++i) {
            if (lj <= i) Hc[i * (i + 1) / 2 + lj] -= Hc[i * (i + 1) / 2 + p] * lpj;
          }
          __syncwarp();
        }
        if (!ok) {
          if (lane == 0) {
            sh_comp_status[c] = STATUS_NOT_PD;
            sh_comp_done[c] = 1;
            sh_comp_rounds[c] = round + 1;
            sh_flags[5] = 1;
          }
          __syncwarp();
          continue;
        }
        float v = (lane < n) ? gc[lane] : 0.0f;
        float gd = v * v;
        for (int o = 16; o > 0; o >>= 1) gd += __shfl_xor_sync(FULL, gd, o);
        if (lane < n) sc[lane] = gc[lane];
        __syncwarp();
        for (int j = 0; j < n; ++j) {
          const float yj = sc[j] / Hc[j * (j + 1) / 2 + j];
          __syncwarp();
          if (lane == 0) sc[j] = yj;
          for (int i = j + 1 + lane; i < n; i += 32) sc[i] -= Hc[i * (i + 1) / 2 + j] * yj;
          __syncwarp();
        }
        for (int j = n - 1; j >= 0; --j) {
          const float sj = sc[j] / Hc[j * (j + 1) / 2 + j];
          __syncwarp();
          if (lane == 0) sc[j] = sj;
          for (int i = lane; i < j; i += 32) sc[i] -= Hc[j * (j + 1) / 2 + i] * sj;
          __syncwarp();
        }
        float sv = (lane < n) ? sc[lane] * gc[lane] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) sv += __shfl_xor_sync(FULL, sv, o);
        const bool finite_ok = isfinite(gd) && isfinite(sv);
        const float grad_r = sqrtf(gd) / scale;
        const float mi_r = 0.5f * sv / scale;
        const bool pass = finite_ok && ((grad_r < comp_tol) || (mi_r < comp_tol));
        const bool last = (round + 1 >= ROUND_CAP);
        if (lane == 0) {
          sh_comp_gd[c] = gd;
          sh_comp_dec[c] = sv;
          sh_comp_rounds[c] = round + 1;
          if (!finite_ok) {
            sh_comp_status[c] = STATUS_NONFINITE;
            sh_comp_done[c] = 1;
            sh_flags[5] = 1;
          } else if (pass || last) {
            sh_comp_done[c] = 1;
            sh_flags[5] = 1;
          }
        }
        __syncwarp();
        if (finite_ok && !pass && !last) {
          // semismooth Newton step on the current active set
          for (int i = lane; i < n; i += 32) sh_q[sb + i] -= sc[i];
        }
        __syncwarp();
      }
    }
    __syncthreads();
    if (DEBUG_EXIT == 4) break;
    int any_open = 0;
    if (tid < nisland && sh_comp_done[tid] == 0) any_open = 1;
    if (__syncthreads_or(any_open) == 0) break;
  }

  // ---------------------------------------------------------------- world certificate
  if (tid == 0) {
    int status = STATUS_CERTIFIED;
    float world_gd = 0.0f;
    float world_dec = 0.0f;
    int max_rounds = 0;
    for (int c = 0; c < nisland; ++c) {
      if (sh_comp_ndof[c] == 0) continue;
      if (sh_comp_status[c] != STATUS_CERTIFIED) status = sh_comp_status[c];
      world_gd += sh_comp_gd[c];
      world_dec += sh_comp_dec[c];
      max_rounds = max(max_rounds, sh_comp_rounds[c]);
    }
    const float grad_r = sqrtf(world_gd) / scale;
    const float mi_r = 0.5f * world_dec / scale;
    const bool pass = isfinite(grad_r) && isfinite(mi_r) && ((grad_r < ctol) || (mi_r < ctol));
    if (status == STATUS_CERTIFIED && !pass) status = STATUS_UNCERTIFIED;
    sh_flags[7] = status;
    status_o[worldid] = status;
    gradient_o[worldid] = grad_r;
    decrement_o[worldid] = mi_r;
    niter_o[worldid] = max_rounds;
    if (status != STATUS_CERTIFIED) {
      stock_o[worldid] = 1;
      atomicAdd(nstock_o, 1);
    } else {
      stock_o[worldid] = 0;
    }
  }
  __syncthreads();

  if (DEBUG_EXIT == 2) return;
  // ---------------------------------------------------------------- phase 3: publication
  for (int g = tid; g < NV; g += BLOCK) {
    const int s = sh_dof_slot[g];
    qacc_o[g] = (s >= 0) ? sh_q[s] : qacc_smooth_p[g];
    qfrc_o[g] = (s >= 0) ? sh_qfrc[s] : 0.0f;
  }
  // Ma = M @ qacc in support.mul_m order
  for (int i = tid; i < NV; i += BLOCK) {
    float acc = 0.0f;
    const int kend = mulm_rowadr_p[i + 1];
    for (int k = mulm_rowadr_p[i]; k < kend; ++k) {
      const int col = mulm_col_p[k];
      const int s = sh_dof_slot[col];
      const float qv = (s >= 0) ? sh_q[s] : qacc_smooth_p[col];
      acc += M_p[mulm_madr_p[k]] * qv;
    }
    Ma_o[i] = acc;
  }
  // force/state for every row from the final Jaref (output-only rows: Jaref = -aref)
  for (int r = tid; r < nefc; r += BLOCK) {
    const int nnz = rownnz_p[r];
    const int adr = rowadr_p[r];
    int cols[NNZ_PREFETCH];
    float vals[NNZ_PREFETCH];
#pragma unroll
    for (int k = 0; k < NNZ_PREFETCH; ++k) {
      cols[k] = (k < nnz) ? colind_p[adr + k] : 0;
      vals[k] = (k < nnz) ? J_p[adr + k] : 0.0f;
    }
    float x = 0.0f;
#pragma unroll
    for (int k = 0; k < NNZ_PREFETCH; ++k) {
      if (k < nnz) {
        const int s = sh_dof_slot[cols[k]];
        if (s >= 0) x += vals[k] * sh_q[s];
      }
    }
    x -= aref_p[r];
    const bool quad = (r < ne) || (x < 0.0f);
    force_o[r] = quad ? -D_p[r] * x : 0.0f;
    state_o[r] = quad ? STATE_QUADRATIC : STATE_SATISFIED;
  }
}
"""


def _render(nv: int, njmax: int, debug_exit: int = 0) -> str:
  replacements = {
    "__BLOCK__": str(BLOCK_DIM),
    "__TILE__": str(TILE_ROWS),
    "__CDOF__": str(COMP_DOF_CAP),
    "__NCDOF__": str(NCDOF_CAP),
    "__NTREE__": str(NTREE_CAP),
    "__NJMAX__": str(njmax),
    "__NV__": str(nv),
    "__ROUND_CAP__": str(ROUND_CAP),
    "__PACKED_TOTAL__": str(PACKED_TOTAL_CAP),
    "__DEBUG_EXIT__": str(debug_exit),
    "__STATE_SATISFIED__": str(int(types.ConstraintState.SATISFIED)),
    "__STATE_QUADRATIC__": str(int(types.ConstraintState.QUADRATIC)),
  }
  source = _NATIVE_TEMPLATE
  for marker, value in replacements.items():
    source = source.replace(marker, value)
  return source


_KERNELS: dict[tuple[int, int, int], wp.Kernel] = {}


def world_solver_kernel(nv: int, njmax: int, debug_exit: int = 0) -> wp.Kernel:
  """Build (and cache) the per-world solver kernel for a model size."""
  key = (nv, njmax, debug_exit)
  kernel = _KERNELS.get(key)
  if kernel is not None:
    return kernel

  @wp.func_native(snippet=_render(nv, njmax, debug_exit))
  def native(
    worldid: int,
    tid: int,
    ntree: int,
    njmax: int,
    nv_scale: int,
    warmstart: int,
    M_elemid: wp.array2d(dtype=int),
    M_mulm_rowadr: wp.array(dtype=int),
    M_mulm_col: wp.array(dtype=int),
    M_mulm_madr: wp.array(dtype=int),
    dof_treeid: wp.array(dtype=int),
    tree_dofadr: wp.array(dtype=int),
    tree_dofnum: wp.array(dtype=int),
    stat_meaninertia: wp.array(dtype=float),
    ctol_in: wp.array(dtype=float),
    ne_in: wp.array(dtype=int),
    nf_in: wp.array(dtype=int),
    nefc_in: wp.array(dtype=int),
    nisland_in: wp.array(dtype=int),
    tree_awake_in: wp.array2d(dtype=int),
    tree_island_in: wp.array2d(dtype=int),
    efc_J_rownnz_in: wp.array2d(dtype=int),
    efc_J_rowadr_in: wp.array2d(dtype=int),
    efc_J_colind_in: wp.array3d(dtype=int),
    efc_J_in: wp.array3d(dtype=float),
    efc_D_in: wp.array2d(dtype=float),
    efc_aref_in: wp.array2d(dtype=float),
    M_in: wp.array2d(dtype=float),
    qacc_smooth_in: wp.array2d(dtype=float),
    qfrc_smooth_in: wp.array2d(dtype=float),
    qacc_warmstart_in: wp.array2d(dtype=float),
    qacc_out: wp.array2d(dtype=float),
    qfrc_constraint_out: wp.array2d(dtype=float),
    efc_force_out: wp.array2d(dtype=float),
    efc_state_out: wp.array2d(dtype=int),
    efc_Ma_out: wp.array2d(dtype=float),
    solver_niter_out: wp.array(dtype=int),
    stock_world_out: wp.array(dtype=int),
    nstock_out: wp.array(dtype=int),
    world_status_out: wp.array(dtype=int),
    world_gradient_out: wp.array(dtype=float),
    world_decrement_out: wp.array(dtype=float),
  ): ...

  @wp.kernel(module="unique", enable_backward=False, grid_stride=False, launch_bounds=(BLOCK_DIM, 2))
  def kernel(
    ntree: int,
    njmax: int,
    nv_scale: int,
    warmstart: int,
    M_elemid: wp.array2d(dtype=int),
    M_mulm_rowadr: wp.array(dtype=int),
    M_mulm_col: wp.array(dtype=int),
    M_mulm_madr: wp.array(dtype=int),
    dof_treeid: wp.array(dtype=int),
    tree_dofadr: wp.array(dtype=int),
    tree_dofnum: wp.array(dtype=int),
    stat_meaninertia: wp.array(dtype=float),
    ctol_in: wp.array(dtype=float),
    ne_in: wp.array(dtype=int),
    nf_in: wp.array(dtype=int),
    nefc_in: wp.array(dtype=int),
    nisland_in: wp.array(dtype=int),
    tree_awake_in: wp.array2d(dtype=int),
    tree_island_in: wp.array2d(dtype=int),
    efc_J_rownnz_in: wp.array2d(dtype=int),
    efc_J_rowadr_in: wp.array2d(dtype=int),
    efc_J_colind_in: wp.array3d(dtype=int),
    efc_J_in: wp.array3d(dtype=float),
    efc_D_in: wp.array2d(dtype=float),
    efc_aref_in: wp.array2d(dtype=float),
    M_in: wp.array2d(dtype=float),
    qacc_smooth_in: wp.array2d(dtype=float),
    qfrc_smooth_in: wp.array2d(dtype=float),
    qacc_warmstart_in: wp.array2d(dtype=float),
    qacc_out: wp.array2d(dtype=float),
    qfrc_constraint_out: wp.array2d(dtype=float),
    efc_force_out: wp.array2d(dtype=float),
    efc_state_out: wp.array2d(dtype=int),
    efc_Ma_out: wp.array2d(dtype=float),
    solver_niter_out: wp.array(dtype=int),
    stock_world_out: wp.array(dtype=int),
    nstock_out: wp.array(dtype=int),
    world_status_out: wp.array(dtype=int),
    world_gradient_out: wp.array(dtype=float),
    world_decrement_out: wp.array(dtype=float),
  ):
    worldid, tid = wp.tid()
    native(
      worldid,
      tid,
      ntree,
      njmax,
      nv_scale,
      warmstart,
      M_elemid,
      M_mulm_rowadr,
      M_mulm_col,
      M_mulm_madr,
      dof_treeid,
      tree_dofadr,
      tree_dofnum,
      stat_meaninertia,
      ctol_in,
      ne_in,
      nf_in,
      nefc_in,
      nisland_in,
      tree_awake_in,
      tree_island_in,
      efc_J_rownnz_in,
      efc_J_rowadr_in,
      efc_J_colind_in,
      efc_J_in,
      efc_D_in,
      efc_aref_in,
      M_in,
      qacc_smooth_in,
      qfrc_smooth_in,
      qacc_warmstart_in,
      qacc_out,
      qfrc_constraint_out,
      efc_force_out,
      efc_state_out,
      efc_Ma_out,
      solver_niter_out,
      stock_world_out,
      nstock_out,
      world_status_out,
      world_gradient_out,
      world_decrement_out,
    )

  _KERNELS[key] = kernel
  return kernel


@dataclasses.dataclass
class WorldSolverContext:
  """Per-world routing and certificate telemetry of the world solver.

  Attributes:
    stock_world: 1 if the world must take the stock solve, else 0    (nworld,)
    nstock: number of worlds routed to the stock solve                (1,)
    status: STATUS_* code per world                                   (nworld,)
    gradient: rescaled gradient norm of the world certificate         (nworld,)
    decrement: rescaled half Newton decrement of the certificate      (nworld,)
  """

  stock_world: wp.array
  nstock: wp.array
  status: wp.array
  gradient: wp.array
  decrement: wp.array


def create_world_solver_context(nworld: int, device=None) -> WorldSolverContext:
  return WorldSolverContext(
    stock_world=wp.zeros(nworld, dtype=int, device=device),
    nstock=wp.zeros(1, dtype=int, device=device),
    status=wp.zeros(nworld, dtype=int, device=device),
    gradient=wp.zeros(nworld, dtype=float, device=device),
    decrement=wp.zeros(nworld, dtype=float, device=device),
  )


def world_solver_unsupported_reason(m: types.Model, d: types.Data) -> str | None:
  """Return why the world solver cannot run on this model, or None."""
  if not (m.opt.enableflags & types.EnableBit.SLEEP):
    return "world solver requires the sleeping (compact) path"
  if m.opt.disableflags & types.DisableBit.ISLAND:
    return "world solver requires islands"
  if m.opt.cone == types.ConeType.ELLIPTIC:
    return "elliptic cones are not supported"
  if m.opt.solver != types.SolverType.NEWTON:
    return "only the Newton solver contract is certified"
  if not m.is_sparse:
    return "world solver reads the sparse constraint Jacobian"
  if m.ntree > NTREE_CAP:
    return f"ntree {m.ntree} exceeds {NTREE_CAP}"
  if m.nflex > 0:
    return "flex is not supported"
  if m.nsensor > 0 and not (m.opt.disableflags & types.DisableBit.SENSOR):
    return "sensors consume efc.force; not validated"
  if d.nvmax_pad > NCDOF_CAP:
    return f"nvmax_pad {d.nvmax_pad} exceeds {NCDOF_CAP}"
  if d.ctol.shape[0] != 1:
    return "compact tolerance missing"
  if d.njmax > 65535:
    return "njmax exceeds the 16-bit row index"
  return None


def launch_world_solver(
  *,
  nworld: int,
  nv: int,
  ntree: int,
  njmax: int,
  nv_scale: int,
  warmstart: bool,
  M_elemid,
  M_mulm_rowadr,
  M_mulm_col,
  M_mulm_madr,
  dof_treeid,
  tree_dofadr,
  tree_dofnum,
  stat_meaninertia,
  ctol,
  ne,
  nf,
  nefc,
  nisland,
  tree_awake,
  tree_island,
  efc_J_rownnz,
  efc_J_rowadr,
  efc_J_colind,
  efc_J,
  efc_D,
  efc_aref,
  M,
  qacc_smooth,
  qfrc_smooth,
  qacc_warmstart,
  qacc,
  qfrc_constraint,
  efc_force,
  efc_state,
  efc_Ma,
  solver_niter,
  ctx: WorldSolverContext,
  debug_exit: int = 0,
):
  """Launch the per-world solver on explicit arrays (zeroes ``ctx.nstock`` first)."""
  ctx.nstock.zero_()
  wp.launch_tiled(
    world_solver_kernel(nv, njmax, debug_exit),
    dim=nworld,
    inputs=[
      ntree,
      njmax,
      nv_scale,
      int(warmstart),
      M_elemid,
      M_mulm_rowadr,
      M_mulm_col,
      M_mulm_madr,
      dof_treeid,
      tree_dofadr,
      tree_dofnum,
      stat_meaninertia,
      ctol,
      ne,
      nf,
      nefc,
      nisland,
      tree_awake,
      tree_island,
      efc_J_rownnz,
      efc_J_rowadr,
      efc_J_colind,
      efc_J,
      efc_D,
      efc_aref,
      M,
      qacc_smooth,
      qfrc_smooth,
      qacc_warmstart,
    ],
    outputs=[
      qacc,
      qfrc_constraint,
      efc_force,
      efc_state,
      efc_Ma,
      solver_niter,
      ctx.stock_world,
      ctx.nstock,
      ctx.status,
      ctx.gradient,
      ctx.decrement,
    ],
    block_dim=BLOCK_DIM,
  )


def world_solve(m: types.Model, d: types.Data, ctx: WorldSolverContext):
  """Run the per-world solver on canonical Data arrays."""
  launch_world_solver(
    nworld=d.nworld,
    nv=m.nv,
    ntree=m.ntree,
    njmax=d.njmax,
    nv_scale=d.nvmax_pad,
    warmstart=not (m.opt.disableflags & types.DisableBit.WARMSTART),
    M_elemid=m.M_elemid,
    M_mulm_rowadr=m.M_mulm_rowadr,
    M_mulm_col=m.M_mulm_col,
    M_mulm_madr=m.M_mulm_madr,
    dof_treeid=m.dof_treeid,
    tree_dofadr=m.tree_dofadr,
    tree_dofnum=m.tree_dofnum,
    stat_meaninertia=m.stat.meaninertia,
    ctol=d.ctol,
    ne=d.ne,
    nf=d.nf,
    nefc=d.nefc,
    nisland=d.nisland,
    tree_awake=d.tree_awake,
    tree_island=d.tree_island,
    efc_J_rownnz=d.efc.J_rownnz,
    efc_J_rowadr=d.efc.J_rowadr,
    efc_J_colind=d.efc.J_colind,
    efc_J=d.efc.J,
    efc_D=d.efc.D,
    efc_aref=d.efc.aref,
    M=d.M,
    qacc_smooth=d.qacc_smooth,
    qfrc_smooth=d.qfrc_smooth,
    qacc_warmstart=d.qacc_warmstart,
    qacc=d.qacc,
    qfrc_constraint=d.qfrc_constraint,
    efc_force=d.efc.force,
    efc_state=d.efc.state,
    efc_Ma=d.efc.Ma,
    solver_niter=d.solver_niter,
    ctx=ctx,
  )
