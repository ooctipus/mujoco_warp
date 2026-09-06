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

Components with up to 16 DOFs are solved warp-synchronously (no block-wide barriers
between rounds): warps pull them from a work-ordered list. Their rows are written once
(phase 0) as dense records to a per-world scratch and then read with one coalesced load
level per round. Components with up to 6 DOFs (free bodies) keep their rows in
registers, reduce the packed 6x6 Hessian with xor-butterfly shuffles and every lane
factors and solves it redundantly in registers. Components with 7..16 DOFs stage 32-row
tiles in per-warp shared memory (resident when the component fits one tile) and
accumulate packed Hessian entries per lane; up to 9 DOFs every lane then gathers the
whole packed Hessian and factors and solves it redundantly in registers (no dependent
shuffle chains), 10..16 DOFs factor with a lane-per-row packed Cholesky in registers.
Components with 17..32 DOFs run on a two-warp group (named barrier) while the other warps
pull the warp-local components: their rows (from a second, wide record scratch) are
staged into the group's pooled arenas (resident when they fit), every group thread
accumulates its live packed Hessian entries over the staged rows into a packed Hessian in
shared memory, and warp 0 factors it with a lane-per-row register Cholesky and solves it.

Every iterate is certified with the history-free clauses of ``solver._solve_done``
(rescaled gradient or half Newton decrement below ``d.ctol`` with
``_rescale(nvmax_pad, meaninertia)``), so a certified world stops at a point at which
the stock solver would also have stopped. Uncertified worlds and worlds that exceed
the compile-time capacities (component DOFs > 32, active DOFs > 128, rows > 1024, more
than 256 rows in 17..32 DOF components, ``nf > 0``) are flagged in ``stock_world`` and
counted in ``nstock`` for the stock fallback.

All reductions run in a fixed order (stable row bucketing, fixed lane-to-row
assignment, xor-butterfly shuffles), so outputs are replay-stable for identical inputs.
"""

from __future__ import annotations

import dataclasses

import warp as wp

from mujoco_warp._src import types

BLOCK_DIM = 128
ROW_CAP = 1024
DENSE_ROW_FLOATS = 20
WIDE_ROW_CAP = 512
WIDE_ROW_FLOATS = 36
LIGHT_DOF_CAP = 6
REGISTER_FACTOR_DOF_CAP = 9
COMP_DOF_CAP = 32
NCDOF_CAP = 128
NTREE_CAP = 32
ROUND_CAP = 10
RESOLVE_ROUND_CAP = 40
RESOLVE_WORLD_CAP = 64
RESIDENT_ROWS_PER_LANE = 2
WIDE_GROUP_WARPS = 2
MIN_BLOCKS_PER_SM = 4

STATUS_CERTIFIED = 0
STATUS_UNCERTIFIED = 1
STATUS_NOT_PD = 3
STATUS_NONFINITE = 4
# capacity routing sub-codes (>= 20): nisland, friction rows, component DOFs, active DOFs,
# rows, row spanning islands, row nnz, wide (17..32 DOF) rows
STATUS_CAPACITY = 20

_NATIVE_TEMPLATE = r"""
{
  constexpr int BLOCK = __BLOCK__;
  constexpr int NWARPS = BLOCK / 32;
  constexpr int ROW_CAP = __ROW_CAP__;
  constexpr int LN = __LN__;                 // light component DOF cap
  constexpr int LP = LN * (LN + 1) / 2;      // packed light Hessian entries
  constexpr int RES = __RES__;               // resident row slots per lane (light path)
  constexpr int M9 = __M9__;                 // mid components up to M9 DOFs factor redundantly in registers
  constexpr int MN = __MN__;                 // component DOF cap (wide components: MNS < n <= MN)
  constexpr int MNS = 16;                    // mid components up to MNS DOFs read dense row records
  constexpr int MP = MNS * (MNS + 1) / 2;    // packed per-warp Hessian entries (n <= MNS)
  constexpr int DR4 = __DR4__;               // float4 per dense row record (MNS + D, aref, coef, F)
  constexpr int WIDE_CAP = __WIDE_CAP__;     // rows of 17..32 DOF components per world (compact index)
  constexpr int WR4 = __WR4__;               // float4 per wide row record (MN + D, aref, coef, F)
  constexpr int WG = __WG__;                 // warps of the wide-component group
  constexpr int MPW = MN * (MN + 1) / 2;     // packed wide Hessian entries
  constexpr int TILE_FLOATS = 32 * (MNS + 4); // per-warp tile: 32 rows x (MNS + D, aref, coef, F)
  constexpr int WARP_FLOATS = TILE_FLOATS + 2 * MP + MN;  // per-warp arena: tile, packed M_c, packed H/L, iterate
  constexpr int NCDOF = __NCDOF__;
  constexpr int NTREE = __NTREE__;
  constexpr int NV = __NV__;
  constexpr int ROUND_CAP = __ROUND_CAP__;
  constexpr int DEBUG_EXIT = __DEBUG_EXIT__;
  constexpr int NNZ_CAP = 16;
  constexpr int NNZ_HALF = NNZ_CAP / 2;
  constexpr int STATE_SATISFIED = __STATE_SATISFIED__;
  constexpr int STATE_QUADRATIC = __STATE_QUADRATIC__;
  constexpr int ROW_NONE = 255;
  constexpr int STATUS_CERTIFIED = 0, STATUS_UNCERTIFIED = 1, STATUS_NOT_PD = 3, STATUS_NONFINITE = 4;
  constexpr int STATUS_CAP_NISLAND = 20, STATUS_CAP_FRICTION = 21, STATUS_CAP_COMP_DOF = 22, STATUS_CAP_NCDOF = 23,
                STATUS_CAP_ROWS = 24, STATUS_CAP_ROW_ISLANDS = 25, STATUS_CAP_ROW_NNZ = 26, STATUS_CAP_WIDE_ROWS = 27;
  static_assert(BLOCK % 32 == 0, "block must be whole warps");
  static_assert(NTREE <= 32, "island bitsets are 32 wide");
  static_assert(LP <= 32, "light Hessian must fit one entry per lane for the M gather");
  static_assert(LN <= 6, "dense row record holds 6 values + D + aref");
  static_assert(LN < M9 && M9 <= MNS, "register-factored mid components read dense row records");
  static_assert(MN <= 32, "mid and wide components use one lane per DOF");
  static_assert(MNS % 4 == 0 && 4 * DR4 == MNS + 4, "dense record: MNS values + D, aref, coef, F");
  static_assert(MN % 4 == 0 && 4 * WR4 == MN + 4, "wide record: MN values + D, aref, coef, F");
  static_assert(2 * 4 >= LN + 2, "light rows use the first two float4 of the record");
  static_assert(LP + LN <= MP, "light M_c/qfrc scratch fits the per-warp H arena");
  static_assert(WG >= 1 && WG <= NWARPS, "the wide group is a subset of the CTA's warps");
  static_assert(TILE_FLOATS % 4 == 0 && MP % 4 == 0 && WARP_FLOATS % 4 == 0, "warp arenas keep float4 alignment");
  static_assert(WG * WARP_FLOATS >= MPW + MN + 4, "the wide group's pooled arenas hold the packed Hessian and a row");
  const unsigned FULL = 0xffffffffu;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const unsigned lanemask_lt = (1u << lane) - 1u;

  // ---- shared memory (block level) ----
  __shared__ unsigned char sh_row_comp[ROW_CAP];
  __shared__ unsigned short sh_row_order[ROW_CAP];
  __shared__ short sh_dof_slot[NV];
  __shared__ unsigned short sh_slot_dof[NCDOF];
  __shared__ unsigned char sh_slot_comp[NCDOF];
  __shared__ float sh_q[NCDOF];
  __shared__ float sh_qfrc[NCDOF];
  __shared__ unsigned char sh_comp_ndof[NTREE];
  __shared__ unsigned char sh_comp_begin[NTREE];
  __shared__ unsigned char sh_comp_status[NTREE];
  __shared__ unsigned char sh_comp_rounds[NTREE];
  __shared__ unsigned char sh_comp_order[NTREE];
  __shared__ unsigned short sh_comp_nrow[NTREE];
  __shared__ unsigned short sh_comp_row_begin[NTREE];
  __shared__ unsigned short sh_comp_wbegin[NTREE];
  __shared__ int sh_comp_cnt[NTREE];
  __shared__ float sh_comp_gd[NTREE];
  __shared__ float sh_comp_dec[NTREE];
  __shared__ int sh_flags[8];
  // ---- shared memory (per-warp arenas; the wide group pools those of warps 0..WG-1) ----
  __shared__ __align__(16) float sh_arena[NWARPS * WARP_FLOATS];
  float* const wtile = sh_arena + warp * WARP_FLOATS;   // 32 staged rows x (MNS + D, aref, coef, F)
  float* const wHw = wtile + TILE_FLOATS;               // mid: packed M_c (+ qfrc_smooth); light: M_c/qfrc scratch
  float* const wHacc = wHw + MP;                        // mid: packed H, then packed L (+ gradient)
  float* const wqw = wHacc + MP;                        // mid: iterate
  // ---- raw pointers and element strides (all 4-byte element types) ----
  const int* const ne_p = reinterpret_cast<const int*>(ne_in.data);
  const int* const nf_p = reinterpret_cast<const int*>(nf_in.data);
  const int* const nefc_p = reinterpret_cast<const int*>(nefc_in.data);
  const int* const nisland_p = reinterpret_cast<const int*>(nisland_in.data);
  const int* const tree_awake_p = reinterpret_cast<const int*>(tree_awake_in.data);
  const int s_tree_awake = tree_awake_in.strides[0] / 4;
  const int* const tree_island_p = reinterpret_cast<const int*>(tree_island_in.data);
  const int s_tree_island = tree_island_in.strides[0] / 4;
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
  // dense row records (DR4 float4 per row), per world
  float4* const dense4_p = reinterpret_cast<float4*>(dense_rows.data) + (size_t)worldid * ROW_CAP * DR4;
  float4* const wide4_p = reinterpret_cast<float4*>(wide_rows.data) + (size_t)worldid * WIDE_CAP * WR4;
  float* const wideM_p = reinterpret_cast<float*>(wide_M.data) + (size_t)worldid * MPW;
  float* const qacc_o = reinterpret_cast<float*>(qacc_out.data) + worldid * (qacc_out.strides[0] / 4);
  float* const qfrc_o = reinterpret_cast<float*>(qfrc_constraint_out.data) + worldid * (qfrc_constraint_out.strides[0] / 4);
  float* const force_o = reinterpret_cast<float*>(efc_force_out.data) + worldid * (efc_force_out.strides[0] / 4);
  int* const state_o = reinterpret_cast<int*>(efc_state_out.data) + worldid * (efc_state_out.strides[0] / 4);
  float* const Ma_o = reinterpret_cast<float*>(efc_Ma_out.data) + worldid * (efc_Ma_out.strides[0] / 4);
  int* const niter_o = reinterpret_cast<int*>(solver_niter_out.data);
  int* const stock_o = reinterpret_cast<int*>(stock_world_out.data);
  int* const nstock_o = reinterpret_cast<int*>(nstock_out.data);
  int* const nstock_total_o = reinterpret_cast<int*>(nstock_total_out.data);
  int* const status_o = reinterpret_cast<int*>(world_status_out.data);
  float* const gradient_o = reinterpret_cast<float*>(world_gradient_out.data);
  float* const decrement_o = reinterpret_cast<float*>(world_decrement_out.data);

  if (DEBUG_EXIT == 9) {
    if (tid == 0) status_o[worldid] = 0;
    return;
  }
  // stock clamps every row loop to njmax (solver.py: wp.min(nefc, njmax))
  const int nefc = min(nefc_p[worldid], njmax);
  const int ne = ne_p[worldid];
  const int nf = nf_p[worldid];
  const int nisland = nisland_p[worldid];
  const float meaninertia = meaninertia_p[worldid % (int)stat_meaninertia.shape[0]];
  const float ctol = ctol_p[0];
  const float scale = meaninertia * (float)nv_scale;

  auto route_to_stock = [&](const int status) {
    if (tid == 0) {
      stock_o[worldid] = 1;
      atomicAdd(nstock_o, 1);
      atomicAdd(nstock_total_o, 1);
      status_o[worldid] = status;
      gradient_o[worldid] = 0.0f;
      decrement_o[worldid] = 0.0f;
    }
  };

  // ---------------------------------------------------------------- phase 0a: components (warp 0)
  if (warp == 0) {
    if (lane < 8) sh_flags[lane] = 0;
    sh_comp_cnt[lane] = 0;
    sh_comp_status[lane] = STATUS_CERTIFIED;
    sh_comp_rounds[lane] = 0;
    sh_comp_gd[lane] = 0.0f;
    sh_comp_dec[lane] = 0.0f;
    sh_comp_nrow[lane] = 0;
    sh_comp_row_begin[lane] = 0;
    int comp = -1;
    int dnum = 0;
    int dadr = 0;
    if (lane < ntree) {
      const int awake = tree_awake_p[worldid * s_tree_awake + lane];
      const int isl = tree_island_p[worldid * s_tree_island + lane];
      dnum = tree_dofnum_p[lane];
      dadr = tree_dofadr_p[lane];
      // ACTIVE predicate of island._compact_dof_layout / _map_compact_dofs
      if (awake == 1 && isl >= 0 && isl < NTREE) comp = isl;
    }
    // island DOF count (lane = island) and the tree's DOF offset inside its island
    int nd = 0;
    int off = 0;
    for (int t2 = 0; t2 < ntree; ++t2) {
      const int c2 = __shfl_sync(FULL, comp, t2);
      const int n2 = __shfl_sync(FULL, dnum, t2);
      if (c2 == lane) nd += n2;
      if (t2 < lane && c2 == comp && comp >= 0) off += n2;
    }
    // exclusive prefix of island DOF counts -> component slot begin
    int incl = nd;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int v = __shfl_up_sync(FULL, incl, o);
      if (lane >= o) incl += v;
    }
    const int begin = incl - nd;
    const int ncdof = __shfl_sync(FULL, incl, 31);
    int status = STATUS_CERTIFIED;
    if (nisland > NTREE) status = STATUS_CAP_NISLAND;
    if (nf > 0) status = STATUS_CAP_FRICTION;
    if (nefc > ROW_CAP) status = STATUS_CAP_ROWS;
    if (__any_sync(FULL, nd > MN)) status = STATUS_CAP_COMP_DOF;
    if (ncdof > NCDOF) status = STATUS_CAP_NCDOF;
    sh_comp_ndof[lane] = (unsigned char)min(nd, 255);
    sh_comp_begin[lane] = (unsigned char)min(begin, 255);
    const int base = __shfl_sync(FULL, begin, (comp >= 0) ? comp : 0) + off;
    if (status == STATUS_CERTIFIED && lane < ntree) {
      if (comp >= 0) {
        for (int j = 0; j < dnum; ++j) {
          const int slot = base + j;
          sh_dof_slot[dadr + j] = (short)slot;
          sh_slot_dof[slot] = (unsigned short)(dadr + j);
          sh_slot_comp[slot] = (unsigned char)comp;
        }
      } else {
        for (int j = 0; j < dnum; ++j) sh_dof_slot[dadr + j] = -1;
      }
    }
    if (lane == 0) {
      sh_flags[0] = status;
      sh_flags[1] = ncdof;
    }
  }
  __syncthreads();
  if (sh_flags[0] != STATUS_CERTIFIED) {
    route_to_stock(sh_flags[0]);
    return;
  }
  const int ncdof = sh_flags[1];

  // ---------------------------------------------------------------- phase 0b: row -> component, dense rows
  // component of a row = island of its first active column; all active columns must agree.
  // Rows of components with <= MNS DOFs are also written as dense records (DR4 float4 per row:
  // j[0..n), then D, aref) so the solve rounds read them with one coalesced load level.
  {
    auto classify = [&](const int r) {
      const int nnz = rownnz_p[r];
      const int adr = rowadr_p[r];
      if (nnz > NNZ_CAP) sh_flags[0] = STATUS_CAP_ROW_NNZ;
      const float Dv = D_p[r];
      const float av = aref_p[r];
      int cols[NNZ_CAP];
      float vals[NNZ_CAP];
#pragma unroll
      for (int k = 0; k < NNZ_CAP; ++k) {
        cols[k] = (k < nnz) ? colind_p[adr + k] : 0;
        vals[k] = (k < nnz) ? J_p[adr + k] : 0.0f;
      }
      int slots[NNZ_CAP];
      int comp = ROW_NONE;
      bool bad = false;
#pragma unroll
      for (int k = 0; k < NNZ_CAP; ++k) {
        const int s = (k < nnz) ? (int)sh_dof_slot[cols[k]] : -1;
        slots[k] = s;
        if (s >= 0) {
          const int c = sh_slot_comp[s];
          if (comp == ROW_NONE) comp = c;
          else if (c != comp) bad = true;
        }
      }
      sh_row_comp[r] = (unsigned char)comp;
      if (comp != ROW_NONE) {
        atomicAdd(&sh_comp_cnt[comp], 1);
        const int nd = sh_comp_ndof[comp];
        if (nd <= MNS) {
          const int sb = sh_comp_begin[comp];
          float jd[MNS];
#pragma unroll
          for (int i = 0; i < MNS; ++i) jd[i] = 0.0f;
          if (nd <= LN) {
#pragma unroll
            for (int k = 0; k < NNZ_CAP; ++k) {
              const int a = slots[k] - sb;
#pragma unroll
              for (int i = 0; i < LN; ++i) jd[i] = (slots[k] >= 0 && a == i) ? vals[k] : jd[i];
            }
            dense4_p[DR4 * r] = make_float4(jd[0], jd[1], jd[2], jd[3]);
            dense4_p[DR4 * r + 1] = make_float4(jd[4], jd[5], Dv, av);
          } else {
#pragma unroll
            for (int k = 0; k < NNZ_CAP; ++k) {
              const int a = slots[k] - sb;
#pragma unroll
              for (int i = 0; i < MNS; ++i) jd[i] = (slots[k] >= 0 && a == i) ? vals[k] : jd[i];
            }
#pragma unroll
            for (int i = 0; i < MNS / 4; ++i) dense4_p[DR4 * r + i] = make_float4(jd[4 * i], jd[4 * i + 1], jd[4 * i + 2], jd[4 * i + 3]);
            dense4_p[DR4 * r + MNS / 4] = make_float4(Dv, av, 0.0f, 0.0f);
          }
        }
      }
      if (bad) sh_flags[0] = STATUS_CAP_ROW_ISLANDS;
    };
    for (int r = tid; r < nefc; r += BLOCK) classify(r);
  }
  __syncthreads();
  if (sh_flags[0] != STATUS_CERTIFIED) {
    route_to_stock(sh_flags[0]);
    return;
  }

  // ---------------------------------------------------------------- phase 0c: component order (warp 0)
  if (warp == 0) {
    const int c = lane;
    const int nrow = (c < nisland) ? sh_comp_cnt[c] : 0;
    const int nd = (c < nisland) ? (int)sh_comp_ndof[c] : 0;
    const bool active = nd > 0;
    // rows keep island order: exclusive prefix of row counts
    int incl = nrow;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int v = __shfl_up_sync(FULL, incl, o);
      if (lane >= o) incl += v;
    }
    const int rbegin = incl - nrow;
    // work-ordered list: heavier components first (LPT-style pull by the warps)
    const int key = active ? (nrow * (nd * (nd + 1) / 2) + 1) : 0;
    int rank = 0;
    for (int c2 = 0; c2 < 32; ++c2) {
      const int k2 = __shfl_sync(FULL, key, c2);
      if (k2 > key || (k2 == key && c2 < lane)) ++rank;
    }
    const unsigned act_mask = __ballot_sync(FULL, active);
    // rows of 17..32 DOF components get compact indices into the wide record scratch
    const bool wide = active && (nd > MNS);
    int inclw = wide ? nrow : 0;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int v = __shfl_up_sync(FULL, inclw, o);
      if (lane >= o) inclw += v;
    }
    const int nwide = __shfl_sync(FULL, inclw, 31);
    sh_comp_nrow[c] = (unsigned short)nrow;
    sh_comp_row_begin[c] = (unsigned short)rbegin;
    sh_comp_wbegin[c] = (unsigned short)(inclw - (wide ? nrow : 0));
    if (active) sh_comp_order[rank] = (unsigned char)c;
    if (lane == 0) {
      sh_flags[2] = __popc(act_mask);
      sh_flags[3] = 0;  // next component to pull
      sh_flags[5] = nwide;
      if (nwide > WIDE_CAP) sh_flags[0] = STATUS_CAP_WIDE_ROWS;
    }
  }
  __syncthreads();
  if (sh_flags[0] != STATUS_CERTIFIED) {
    route_to_stock(sh_flags[0]);
    return;
  }
  const int ncomp_active = sh_flags[2];
  const int nwide = sh_flags[5];

  // ---------------------------------------------------------------- phase 0d: stable bucketing
  for (int c = warp; c < nisland; c += NWARPS) {
    const int nrow = sh_comp_nrow[c];
    if (nrow == 0) continue;
    const int rb = sh_comp_row_begin[c];
    int fill = 0;
    for (int base = 0; base < nefc; base += 32) {
      const int r = base + lane;
      const bool mine = (r < nefc) && (sh_row_comp[r] == c);
      const unsigned m = __ballot_sync(FULL, mine);
      if (mine) sh_row_order[rb + fill + __popc(m & lanemask_lt)] = (unsigned short)r;
      fill += __popc(m);
    }
  }
  __syncthreads();
  // ---------------------------------------------------------------- phase 0e: wide row records
  // rows of 17..32 DOF components as dense records (j[0..MN), D, aref) at their compact index
  if (nwide > 0) {
    for (int i = tid; i < nwide; i += BLOCK) {
      int c = 0;
      for (; c < nisland; ++c) {
        if ((int)sh_comp_ndof[c] > MNS && i >= (int)sh_comp_wbegin[c] && i < (int)sh_comp_wbegin[c] + (int)sh_comp_nrow[c]) break;
      }
      const int r = sh_row_order[(int)sh_comp_row_begin[c] + (i - (int)sh_comp_wbegin[c])];
      const int sb = sh_comp_begin[c];
      const int nnz = rownnz_p[r];
      const int adr = rowadr_p[r];
      float jd[MN];
#pragma unroll
      for (int a = 0; a < MN; ++a) jd[a] = 0.0f;
#pragma unroll
      for (int k = 0; k < NNZ_CAP; ++k) {
        const int col = (k < nnz) ? colind_p[adr + k] : 0;
        const float val = (k < nnz) ? J_p[adr + k] : 0.0f;
        const int s = (k < nnz) ? (int)sh_dof_slot[col] : -1;
        const int a = s - sb;
#pragma unroll
        for (int i2 = 0; i2 < MN; ++i2) jd[i2] = (s >= 0 && a == i2) ? val : jd[i2];
      }
#pragma unroll
      for (int q4 = 0; q4 < MN / 4; ++q4) wide4_p[WR4 * i + q4] = make_float4(jd[4 * q4], jd[4 * q4 + 1], jd[4 * q4 + 2], jd[4 * q4 + 3]);
      wide4_p[WR4 * i + MN / 4] = make_float4(D_p[r], aref_p[r], 0.0f, 0.0f);
    }
    __syncthreads();
  }
  if (DEBUG_EXIT == 1) {
    if (tid == 0) status_o[worldid] = 0;
    return;
  }

  const float comp_tol = ctol / (float)max(ncomp_active, 1);
  const int round_cap = ((DEBUG_EXIT & 15) == 3) ? 0 : (((DEBUG_EXIT & 15) == 4) ? 1 : ROUND_CAP);
  constexpr bool DBG_NO_STAGE = (DEBUG_EXIT & 16) != 0;
  constexpr bool DBG_NO_ACC = (DEBUG_EXIT & 32) != 0;
  constexpr bool DBG_NO_FACTOR = (DEBUG_EXIT & 64) != 0;

  // ---------------------------------------------------------------- light path (n <= LN, registers)
  // One warp per component. Rows are dense records in registers (RES per lane) plus streamed
  // batches of 32; the packed 6x6 Hessian is reduced with xor-butterfly shuffles and every lane
  // factors and solves it redundantly in registers.
  auto light_component = [&](const int c) {
    const int n = sh_comp_ndof[c];
    const int sb = sh_comp_begin[c];
    const int nrow = sh_comp_nrow[c];
    const int rb = sh_comp_row_begin[c];
    float* const Mc = wHw;           // packed M_c (LP entries)
    float* const qs = Mc + LP;       // qfrc_smooth on the component slots
    if (lane < LP) {
      int a = 0;
      while ((a + 1) * (a + 2) / 2 <= lane) ++a;
      const int b = lane - a * (a + 1) / 2;
      float v = 0.0f;
      if (a < n) {
        const int ga = sh_slot_dof[sb + a];
        const int gb = sh_slot_dof[sb + b];
        const int elemid = M_elemid_p[max(ga, gb) * s_M_elemid + min(ga, gb)];
        if (elemid >= 0) v = M_p[elemid];
      }
      Mc[lane] = v;
    }
    if (lane < LN) qs[lane] = (lane < n) ? qfrc_smooth_p[sh_slot_dof[sb + lane]] : 0.0f;
    float q[LN];
#pragma unroll
    for (int i = 0; i < LN; ++i) {
      float v = 0.0f;
      if (i < n) {
        const int g = sh_slot_dof[sb + i];
        v = warmstart ? warm_p[g] : qacc_smooth_p[g];
      }
      q[i] = v;
    }
    // resident rows (fixed lane assignment: row index i*32 + lane)
    int rr[RES];
    float jd[RES][LN];
    float Dd[RES];
    float ad[RES];
#pragma unroll
    for (int i = 0; i < RES; ++i) {
      const int idx = i * 32 + lane;
      const int r = (idx < nrow) ? (int)sh_row_order[rb + idx] : -1;
      rr[i] = r;
      float4 v0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      float4 v1 = v0;
      if (r >= 0) {
        v0 = dense4_p[DR4 * r];
        v1 = dense4_p[DR4 * r + 1];
      }
      jd[i][0] = v0.x; jd[i][1] = v0.y; jd[i][2] = v0.z; jd[i][3] = v0.w;
      jd[i][4] = v1.x; jd[i][5] = v1.y;
      Dd[i] = v1.z;
      ad[i] = v1.w;
    }
    __syncwarp();
    float Hp[LP];
    float g[LN];
    float gd = 0.0f;
    float sv = 0.0f;
    int status = STATUS_CERTIFIED;
    int rounds = 0;
    for (int round = 0; round < round_cap; ++round) {
      __syncwarp();  // keeps the M_c / qfrc reads below per round (register pressure)
#pragma unroll
      for (int e = 0; e < LP; ++e) Hp[e] = 0.0f;
#pragma unroll
      for (int a = 0; a < LN; ++a) g[a] = 0.0f;
      // resident rows: classify at q, accumulate H += D j j^T and J^T f (QUADRATIC rows only)
#pragma unroll
      for (int i = 0; i < RES; ++i) {
        float x = -ad[i];
#pragma unroll
        for (int k = 0; k < LN; ++k) x += jd[i][k] * q[k];
        // _eval_constraint: equality -> QUADRATIC; limit/contact -> QUADRATIC iff jaref < 0
        const bool quad = (rr[i] >= 0) && ((rr[i] < ne) || (x < 0.0f));
        const float coef = quad ? Dd[i] : 0.0f;
        const float f = -coef * x;
#pragma unroll
        for (int a = 0; a < LN; ++a) {
          const float ca = coef * jd[i][a];
          g[a] += f * jd[i][a];
#pragma unroll
          for (int b = 0; b < LN; ++b) {
            if (b <= a) Hp[a * (a + 1) / 2 + b] += ca * jd[i][b];
          }
        }
      }
      // streamed rows (components with more than RES*32 rows)
      for (int base = RES * 32; base < nrow; base += 32) {
        const int idx = base + lane;
        const int r = (idx < nrow) ? (int)sh_row_order[rb + idx] : -1;
        float4 v0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 v1 = v0;
        if (r >= 0) {
          v0 = dense4_p[DR4 * r];
          v1 = dense4_p[DR4 * r + 1];
        }
        float js[LN];
        js[0] = v0.x; js[1] = v0.y; js[2] = v0.z; js[3] = v0.w; js[4] = v1.x; js[5] = v1.y;
        float x = -v1.w;
#pragma unroll
        for (int k = 0; k < LN; ++k) x += js[k] * q[k];
        const bool quad = (r >= 0) && ((r < ne) || (x < 0.0f));
        const float coef = quad ? v1.z : 0.0f;
        const float f = -coef * x;
#pragma unroll
        for (int a = 0; a < LN; ++a) {
          const float ca = coef * js[a];
          g[a] += f * js[a];
#pragma unroll
          for (int b = 0; b < LN; ++b) {
            if (b <= a) Hp[a * (a + 1) / 2 + b] += ca * js[b];
          }
        }
      }
      // fixed-order all-reduce: every lane ends with the totals
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
#pragma unroll
        for (int e = 0; e < LP; ++e) Hp[e] += __shfl_xor_sync(FULL, Hp[e], o);
#pragma unroll
        for (int a = 0; a < LN; ++a) g[a] += __shfl_xor_sync(FULL, g[a], o);
      }
      // s = M q - qfrc_smooth - J^T f (the gradient; solved in place below);
      // Hp <- M + J^T D J (identity padding beyond n)
      float s[LN];
#pragma unroll
      for (int a = 0; a < LN; ++a) {
        float mq = 0.0f;
#pragma unroll
        for (int b = 0; b < LN; ++b) {
          const int hi = (a > b) ? a : b;
          const int lo = (a > b) ? b : a;
          mq += Mc[hi * (hi + 1) / 2 + lo] * q[b];
        }
        s[a] = (a < n) ? (mq - qs[a] - g[a]) : 0.0f;
#pragma unroll
        for (int b = 0; b < LN; ++b) {
          if (b <= a) {
            const int e = a * (a + 1) / 2 + b;
            Hp[e] = (a < n) ? (Mc[e] + Hp[e]) : ((a == b) ? 1.0f : 0.0f);
          }
        }
      }
      gd = 0.0f;
#pragma unroll
      for (int i = 0; i < LN; ++i) gd += s[i] * s[i];
      // packed Cholesky in registers (all lanes redundantly); the diagonal stores 1/L[p][p]
      bool ok = true;
#pragma unroll
      for (int p = 0; p < LN; ++p) {
        const int rp = p * (p + 1) / 2;
        float d = Hp[rp + p];
#pragma unroll
        for (int k = 0; k < LN; ++k) {
          if (k < p) d -= Hp[rp + k] * Hp[rp + k];
        }
        ok = ok && (d > 0.0f) && isfinite(d);
        const float inv = rsqrtf(d);
        Hp[rp + p] = inv;
#pragma unroll
        for (int i = 0; i < LN; ++i) {
          if (i > p) {
            const int ri = i * (i + 1) / 2;
            float v = Hp[ri + p];
#pragma unroll
            for (int k = 0; k < LN; ++k) {
              if (k < p) v -= Hp[ri + k] * Hp[rp + k];
            }
            Hp[ri + p] = v * inv;
          }
        }
      }
      rounds = round + 1;
      if (!ok) {
        status = STATUS_NOT_PD;
        break;
      }
      // forward y = L^-1 grad (in place); s^T grad = |y|^2; backward s = L^-T y (in place)
#pragma unroll
      for (int i = 0; i < LN; ++i) {
        const int ri = i * (i + 1) / 2;
        float y = s[i];
#pragma unroll
        for (int k = 0; k < LN; ++k) {
          if (k < i) y -= Hp[ri + k] * s[k];
        }
        s[i] = y * Hp[ri + i];
      }
      sv = 0.0f;
#pragma unroll
      for (int i = 0; i < LN; ++i) sv += s[i] * s[i];
#pragma unroll
      for (int i = LN - 1; i >= 0; --i) {
        float y = s[i];
#pragma unroll
        for (int k = 0; k < LN; ++k) {
          if (k > i) y -= Hp[k * (k + 1) / 2 + i] * s[k];
        }
        s[i] = y * Hp[i * (i + 1) / 2 + i];
      }
      const bool finite_ok = isfinite(gd) && isfinite(sv);
      const float grad_r = sqrtf(gd) / scale;
      const float mi_r = 0.5f * sv / scale;
      const bool pass = finite_ok && ((grad_r < comp_tol) || (mi_r < comp_tol));
      const bool last = (round + 1 >= round_cap);
      if (!finite_ok) {
        status = STATUS_NONFINITE;
        break;
      }
      if (pass || last) break;
      // semismooth Newton step on the current active set
#pragma unroll
      for (int i = 0; i < LN; ++i) q[i] -= s[i];
    }
    // force/state for every row of the component at the final iterate
#pragma unroll
    for (int i = 0; i < RES; ++i) {
      if (rr[i] >= 0) {
        float x = -ad[i];
#pragma unroll
        for (int k = 0; k < LN; ++k) x += jd[i][k] * q[k];
        const bool quad = (rr[i] < ne) || (x < 0.0f);
        force_o[rr[i]] = quad ? -Dd[i] * x : 0.0f;
        state_o[rr[i]] = quad ? STATE_QUADRATIC : STATE_SATISFIED;
      }
    }
    for (int base = RES * 32; base < nrow; base += 32) {
      const int idx = base + lane;
      if (idx < nrow) {
        const int r = sh_row_order[rb + idx];
        const float4 v0 = dense4_p[DR4 * r];
        const float4 v1 = dense4_p[DR4 * r + 1];
        const float x = v0.x * q[0] + v0.y * q[1] + v0.z * q[2] + v0.w * q[3] + v1.x * q[4] + v1.y * q[5] - v1.w;
        const bool quad = (r < ne) || (x < 0.0f);
        force_o[r] = quad ? -v1.z * x : 0.0f;
        state_o[r] = quad ? STATE_QUADRATIC : STATE_SATISFIED;
      }
    }
    if (lane < n) {
      // select (not index) so q/g stay in registers
      float qv = 0.0f;
      float gv = 0.0f;
#pragma unroll
      for (int i = 0; i < LN; ++i) {
        qv = (lane == i) ? q[i] : qv;
        gv = (lane == i) ? g[i] : gv;
      }
      sh_q[sb + lane] = qv;
      sh_qfrc[sb + lane] = gv;
    }
    if (lane == 0) {
      sh_comp_status[c] = (unsigned char)status;
      sh_comp_rounds[c] = (unsigned char)rounds;
      sh_comp_gd[c] = gd;
      sh_comp_dec[c] = sv;
    }
    __syncwarp();
  };

  // ---------------------------------------------------------------- mid path (LN < n <= MNS, one warp)
  // Rows are staged row-major into the warp's shared tile (TS floats per row: j[0..MNS), D,
  // aref, coef, F); the tile stays staged across rounds when the component fits it. The lane
  // owns the packed Hessian entries E = lane + 32k (kept as (a << 8) | b) and accumulates
  // H[a][b] += coef j_a j_b over the tile rows; lane a accumulates the gradient entry a. The
  // packed M_c lives in the warp's M_c arena, the packed H (then L) in its H arena. Two factor
  // variants share the staging and the accumulation:
  //   REG (n <= M9): every lane gathers the whole packed Hessian from the arena and factors and
  //     solves it redundantly in registers (no dependent shuffle chains);
  //   lane-per-row (n <= MNS): lane a holds row a of H / L in registers, right-looking packed
  //     Cholesky with chunked straight-line shuffles, backward solve from the packed L in shared
  //     memory.
  // Both apply the same operations in the same order, so the iterate does not depend on the
  // variant.
  auto mid_component = [&](auto tag, const int c) {
    constexpr int MNL = (int)sizeof(*tag);            // Hessian dimension cap of the variant
    constexpr bool REG = MNL <= M9;
    constexpr int TS = MNS + 4;                       // tile row stride
    constexpr int TB = 32;                            // rows per tile
    constexpr int MPL = MNL * (MNL + 1) / 2;          // packed entries of the variant
    constexpr int EPLL = (MPL + 31) / 32;             // packed entries per lane
    constexpr int CH = 4;                             // lane-per-row: straight-line shuffle steps per chunk
    static_assert(REG || MNL % CH == 0, "the lane-per-row factor chunks tile the row");
    static_assert(MNL == M9 || MNL == MNS, "mid variants: register factor up to M9, lane-per-row up to MNS");
    static_assert(TB * TS <= TILE_FLOATS && 4 * DR4 == TS, "tile rows are dense record copies");
    static_assert(MPL <= MP && (!REG || MPL + MNL <= MP), "packed M_c / H (and the REG vectors) fit the warp arenas");
    const int n = sh_comp_ndof[c];
    const int sb = sh_comp_begin[c];
    const int nrow = sh_comp_nrow[c];
    const int rb = sh_comp_row_begin[c];
    const bool own_row = lane < n;
    const bool resident = nrow <= TB;
    const int rowp = lane * (lane + 1) / 2;
    const int np = n * (n + 1) / 2;
    float* const tile = wtile;
    float* const Mpk = wHw;                           // packed M_c
    float* const qsm = Mpk + MPL;                     // REG: qfrc_smooth on the component slots
    float* const Hpk = wHacc;                         // packed H (lane entries), then packed L (lane-per-row)
    float* const gsh = Hpk + MPL;                     // REG: gradient J^T f per DOF
    float* const qc = wqw;                            // iterate (REG: only the initial value)
    // packed M_c: the lane's entries E = lane + 32k, gathered with one dependent load level
    int eab[EPLL];
#pragma unroll
    for (int k = 0; k < EPLL; ++k) {
      const int E = lane + 32 * k;
      int a = (int)((sqrtf(8.0f * (float)E + 1.0f) - 1.0f) * 0.5f);
      while ((a + 1) * (a + 2) / 2 <= E) ++a;
      while (a * (a + 1) / 2 > E) --a;
      const int b = E - a * (a + 1) / 2;
      eab[k] = (a << 8) | b;
      if (E < np) {
        const int ga_ = sh_slot_dof[sb + a];
        const int gb = sh_slot_dof[sb + b];
        const int elemid = M_elemid_p[max(ga_, gb) * s_M_elemid + min(ga_, gb)];
        Mpk[E] = (elemid >= 0) ? M_p[elemid] : 0.0f;
      }
    }
    float qfs = 0.0f;
    if (own_row) {
      const int g = sh_slot_dof[sb + lane];
      qfs = qfrc_smooth_p[g];
      qc[lane] = warmstart ? warm_p[g] : qacc_smooth_p[g];
      if (REG) qsm[lane] = qfs;
    }
    __syncwarp();
    // REG: the iterate is held redundantly in registers (zero beyond n)
    float q[MNL];
#pragma unroll
    for (int i = 0; i < MNL; ++i) q[i] = (REG && i < n) ? qc[i] : 0.0f;
    // stage rows [tb, tb+rows): copy the dense record (j[0..MNS), D, aref) into tile row t;
    // streamed batches classify straight from the loaded registers (coef, F fill the record)
    auto stage_batch = [&](const int tb, const int rows, const bool classify) {
      const int t = lane;
      if (t < rows) {
        const int r = sh_row_order[rb + tb + t];
        float4 w[DR4];
#pragma unroll
        for (int i = 0; i < DR4; ++i) w[i] = dense4_p[DR4 * r + i];
        if (classify) {
          float x = -w[DR4 - 1].y;
#pragma unroll
          for (int b = 0; b < MNL; ++b) {
            if (b >= n) break;
            const float4 wb = w[b >> 2];
            const float vb = ((b & 3) == 0) ? wb.x : (((b & 3) == 1) ? wb.y : (((b & 3) == 2) ? wb.z : wb.w));
            x += vb * (REG ? q[b] : qc[b]);
          }
          const bool quad = (r < ne) || (x < 0.0f);
          const float Dv = w[DR4 - 1].x;
          const float f = quad ? -Dv * x : 0.0f;
          w[DR4 - 1].z = quad ? Dv : 0.0f;
          w[DR4 - 1].w = f;
          force_o[r] = f;
          state_o[r] = quad ? STATE_QUADRATIC : STATE_SATISFIED;
        }
        float4* const row4 = reinterpret_cast<float4*>(tile + t * TS);
#pragma unroll
        for (int i = 0; i < DR4; ++i) row4[i] = w[i];
      }
      __syncwarp();
    };
    // classify tile rows at the current iterate (lane = row): coef = D on QUADRATIC rows, F = -D x;
    // force/state are published every round (the final round wins)
    auto classify_tile = [&](const int tb, const int rows) {
      if (lane < rows) {
        const int r = sh_row_order[rb + tb + lane];
        float* const row = tile + lane * TS;
        float x = -row[MNS + 1];
#pragma unroll
        for (int a = 0; a < MNL; ++a) {
          if (a >= n) break;
          x += row[a] * (REG ? q[a] : qc[a]);
        }
        const bool quad = (r < ne) || (x < 0.0f);
        const float Dv = row[MNS];
        const float f = quad ? -Dv * x : 0.0f;
        row[MNS + 2] = quad ? Dv : 0.0f;
        row[MNS + 3] = f;
        force_o[r] = f;
        state_o[r] = quad ? STATE_QUADRATIC : STATE_SATISFIED;
      }
      __syncwarp();
    };
    // xor-butterfly sum over the 32 lanes in the fixed offset order 16, 8, 4, 2, 1 (every lane ends
    // with the same total); REG replays the same association on a register vector padded with zeros
    auto butterfly = [&](float v) {
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(FULL, v, o);
      return v;
    };
    auto butterfly_reg = [&](const float* v) {
      float t[32];
#pragma unroll
      for (int i = 0; i < 32; ++i) t[i] = (i < MNL) ? v[i] : 0.0f;
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        float u[32];
#pragma unroll
        for (int i = 0; i < 32; ++i) u[i] = t[i] + t[i ^ o];
#pragma unroll
        for (int i = 0; i < 32; ++i) t[i] = u[i];
      }
      return t[0];
    };
    stage_batch(0, min(TB, nrow), false);
    float gd = 0.0f;
    float sv = 0.0f;
    int status = STATUS_CERTIFIED;
    int rounds = 0;
    float ga = 0.0f;
    for (int round = 0; round < round_cap; ++round) {
      // packed entries start from M_c; lane a accumulates the gradient entry a
      float hp[EPLL];
#pragma unroll
      for (int k = 0; k < EPLL; ++k) {
        const int E = lane + 32 * k;
        hp[k] = 0.0f;
        if (E < np) hp[k] = Mpk[E];
      }
      ga = 0.0f;
      const int trow = own_row ? lane : 0;
      for (int tb = 0; tb < nrow; tb += TB) {
        const int rows = min(TB, nrow - tb);
        if (!resident && !DBG_NO_STAGE) stage_batch(tb, rows, true);
        if (resident) classify_tile(tb, rows);
        // H[a][b] += coef j_a j_b over the tile rows (SATISFIED rows carry coef = 0)
#pragma unroll 4
        for (int tt = 0; tt < (DBG_NO_ACC ? 0 : rows); ++tt) {
          const float* const row = tile + tt * TS;
          const float cf = row[MNS + 2];
          const float fL = row[MNS + 3];
#pragma unroll
          for (int k = 0; k < EPLL; ++k) hp[k] += cf * row[eab[k] >> 8] * row[eab[k] & 255];
          ga += fL * row[trow];
        }
        __syncwarp();
      }
      // packed entries -> arena
#pragma unroll
      for (int k = 0; k < EPLL; ++k) {
        const int E = lane + 32 * k;
        if (E < np) Hpk[E] = hp[k];
      }
      if (REG && own_row) gsh[lane] = ga;
      __syncwarp();
      bool ok = true;
      float s_own = 0.0f;
      float gv = 0.0f;
      if (REG) {
        // every lane: the whole packed H (identity padding beyond n) and the gradient
        // g = M q - qfrc_smooth - J^T f; the diagonal of the factor stores 1/L[p][p]
        float H[MPL];
#pragma unroll
        for (int a = 0; a < MNL; ++a) {
#pragma unroll
          for (int b = 0; b <= a; ++b) {
            const int e = a * (a + 1) / 2 + b;
            H[e] = (a < n) ? Hpk[e] : ((a == b) ? 1.0f : 0.0f);
          }
        }
        float g[MNL];
        float s[MNL];
#pragma unroll
        for (int a = 0; a < MNL; ++a) {
          float mq = 0.0f;
#pragma unroll
          for (int b = 0; b < MNL; ++b) {
            if (b >= n) break;
            const int hi = (a > b) ? a : b;
            const int lo = (a > b) ? b : a;
            mq += Mpk[hi * (hi + 1) / 2 + lo] * q[b];
          }
          g[a] = (a < n) ? (mq - qsm[a] - gsh[a]) : 0.0f;
          s[a] = g[a] * g[a];
        }
        gd = butterfly_reg(s);
        // right-looking packed Cholesky in registers (same operation order as the lane-per-row factor)
#pragma unroll
        for (int p = 0; p < MNL; ++p) {
          if (p >= n || DBG_NO_FACTOR) break;
          const int rp = p * (p + 1) / 2;
          const float d = H[rp + p];
          ok = ok && (d > 0.0f) && isfinite(d);
          const float inv = rsqrtf(d);
          H[rp + p] = inv;
#pragma unroll
          for (int i = p + 1; i < MNL; ++i) {
            if (i >= n) break;
            H[i * (i + 1) / 2 + p] *= inv;  // L[i][p]
          }
#pragma unroll
          for (int i = p + 1; i < MNL; ++i) {
            if (i >= n) break;
            const int ri = i * (i + 1) / 2;
#pragma unroll
            for (int j = p + 1; j <= i; ++j) H[ri + j] -= H[ri + p] * H[j * (j + 1) / 2 + p];
          }
        }
        rounds = round + 1;
        if (!ok) {
          status = STATUS_NOT_PD;
          break;
        }
        // forward y = L^-1 g (in place), backward s = L^-T y (in place, descending), sv = s . g
#pragma unroll
        for (int i = 0; i < MNL; ++i) {
          if (i >= n || DBG_NO_FACTOR) break;
          const int ri = i * (i + 1) / 2;
          float y = g[i];
#pragma unroll
          for (int k = 0; k < i; ++k) y -= H[ri + k] * s[k];
          s[i] = y * H[ri + i];
        }
#pragma unroll
        for (int i = MNL - 1; i >= 0; --i) {
          if (i >= n || DBG_NO_FACTOR) continue;
          const int ri = i * (i + 1) / 2;
          float y = s[i];
#pragma unroll
          for (int j = MNL - 1; j > i; --j) {
            if (j >= n) continue;
            y -= H[j * (j + 1) / 2 + i] * s[j];
          }
          s[i] = y * H[ri + i];
        }
        float sg[MNL];
#pragma unroll
        for (int i = 0; i < MNL; ++i) sg[i] = (i < n) ? s[i] * g[i] : 0.0f;
        sv = butterfly_reg(sg);
        const bool finite_ok = isfinite(gd) && isfinite(sv);
        const float grad_r = sqrtf(gd) / scale;
        const float mi_r = 0.5f * sv / scale;
        const bool pass = finite_ok && ((grad_r < comp_tol) || (mi_r < comp_tol));
        const bool last = (round + 1 >= round_cap);
        if (!finite_ok) {
          status = STATUS_NONFINITE;
          break;
        }
        if (pass || last) break;
        // semismooth Newton step on the current active set
#pragma unroll
        for (int i = 0; i < MNL; ++i) q[i] -= s[i];
      } else {
        // lane a owns row a of H in registers (zero beyond the row and beyond n)
        float Hrow[MNL];
#pragma unroll
        for (int b = 0; b < MNL; ++b) Hrow[b] = (own_row && b <= lane) ? Hpk[rowp + b] : 0.0f;
        __syncwarp();
        float mq = 0.0f;
        if (own_row) {
#pragma unroll 4
          for (int b = 0; b < n; ++b) {
            const int hi = max(lane, b);
            const int lo = min(lane, b);
            mq += Mpk[hi * (hi + 1) / 2 + lo] * qc[b];
          }
        }
        gv = own_row ? (mq - qfs - ga) : 0.0f;
        // packed Cholesky, lane = row, right-looking; inv_own = 1/L[lane][lane]. The column
        // shuffles run in chunks of CH straight-line steps (pipelined, bounded register pressure);
        // every loop has a constant trip count so Hrow stays in registers
        float inv_own = 0.0f;
#pragma unroll
        for (int p = 0; p < MNL; ++p) {
          if (p >= n || DBG_NO_FACTOR) break;
          const float d = __shfl_sync(FULL, Hrow[p], p);
          ok = ok && (d > 0.0f) && isfinite(d);
          const float inv = rsqrtf(d);
          if (lane == p) inv_own = inv;
          if (lane > p) Hrow[p] *= inv;  // L[lane][p]
          const float lp = Hrow[p];
#pragma unroll
          for (int jb = 0; jb < MNL; jb += CH) {
            if (jb + CH <= p + 1) continue;  // chunk at or before the pivot (folded)
            if (jb >= n) break;
#pragma unroll
            for (int j = jb; j < jb + CH && j < MNL; ++j) {
              if (j > p) {
                const float ljp = __shfl_sync(FULL, lp, j);
                if (j < n && lane >= j) Hrow[j] -= lp * ljp;
              }
            }
          }
        }
        rounds = round + 1;
        if (!ok) {
          status = STATUS_NOT_PD;
          break;
        }
        gd = butterfly(gv * gv);
        // forward: y = L^-1 grad (lane k publishes y_k at step k)
        float acc = gv;
        float y_own = 0.0f;
#pragma unroll
        for (int k = 0; k < MNL; ++k) {
          if (k >= n || DBG_NO_FACTOR) break;
          const float yk = __shfl_sync(FULL, acc * inv_own, k);
          if (lane == k) y_own = yk;
          if (lane > k) acc -= Hrow[k] * yk;
        }
        // packed L to shared (own row) for the backward solve
#pragma unroll
        for (int b = 0; b < MNL; ++b) {
          if (own_row && b <= lane) Hpk[rowp + b] = Hrow[b];
        }
        __syncwarp();
        // backward: s = L^-T y (L[j][lane] from the shared packed copy, loaded CH steps ahead of
        // the shuffle chain)
        acc = y_own;
#pragma unroll
        for (int jb = MNL - CH; jb >= 0; jb -= CH) {
          if (jb >= n) continue;  // chunk beyond n
          float lj[CH];
#pragma unroll
          for (int i = 0; i < CH; ++i) {
            const int j = jb + i;
            lj[i] = (j < n && own_row && lane < j) ? Hpk[j * (j + 1) / 2 + lane] : 0.0f;
          }
#pragma unroll
          for (int i = CH - 1; i >= 0; --i) {
            const int j = jb + i;
            const bool live = (j < n) && !DBG_NO_FACTOR;
            const float sj = __shfl_sync(FULL, acc * inv_own, j);
            if (live && lane == j) s_own = sj;
            if (live && lane < j) acc -= lj[i] * sj;
          }
        }
        sv = butterfly(s_own * gv);
        const bool finite_ok = isfinite(gd) && isfinite(sv);
        const float grad_r = sqrtf(gd) / scale;
        const float mi_r = 0.5f * sv / scale;
        const bool pass = finite_ok && ((grad_r < comp_tol) || (mi_r < comp_tol));
        const bool last = (round + 1 >= round_cap);
        if (!finite_ok) {
          status = STATUS_NONFINITE;
          break;
        }
        if (pass || last) break;
        // semismooth Newton step on the current active set
        __syncwarp();
        if (own_row) qc[lane] -= s_own;
        __syncwarp();
      }
    }
    // force/state were stored by the last round's classification (the final iterate)
    if (own_row) {
      float qv = qc[lane];
      if (REG) {
#pragma unroll
        for (int i = 0; i < MNL; ++i) qv = (lane == i) ? q[i] : qv;
      }
      sh_q[sb + lane] = qv;
      sh_qfrc[sb + lane] = ga;
    }
    if (lane == 0) {
      sh_comp_status[c] = (unsigned char)status;
      sh_comp_rounds[c] = (unsigned char)rounds;
      sh_comp_gd[c] = gd;
      sh_comp_dec[c] = sv;
    }
    __syncwarp();
  };

  // ---------------------------------------------------------------- wide path (MNS < n <= MN, WG warps)
  // Warps 0..WG-1 form a group (named barrier 1) that solves the wide components while the other
  // warps pull warp-local components. The group's warp arenas are pooled: the packed Hessian (M_c
  // at the start of a round, then H, then L) at the front, staged rows (WR4 float4 each) behind it;
  // components whose rows fit stay staged across rounds and are reclassified in place. Group
  // thread t owns the packed entries E = t + WT k below np (their M_c values in a per-world
  // scratch, reloaded every round in flight with the row loads) and sweeps every staged row, so
  // H is complete without a reduction. Warp 0 owns lane = DOF: M_c q from the arena before the sweep, then the
  // right-looking packed Cholesky with lane = row in registers (straight-line shuffles, as in the
  // mid path), the shuffle solves, the certificate and the step.
  auto wide_component = [&](const int c) {
    constexpr int WT = 32 * WG;                                 // group threads
    constexpr int TS = MN + 4;                                  // staged row stride
    constexpr int POOL = WG * WARP_FLOATS - MPW;                // pooled floats behind the packed Hessian
    constexpr int WCH = (POOL / TS) < WT ? (POOL / TS) : WT;    // rows staged per pass
    constexpr int EPT = (MPW + WT - 1) / WT;                    // packed entries per thread (cap)
    constexpr int CH = 4;                                       // straight-line shuffle steps per chunk
    static_assert(4 * WR4 == TS && MN % CH == 0, "staged rows are wide record copies; chunks tile the row");
    static_assert(MPW % 4 == 0 && WCH >= 1, "the pool holds at least one float4-aligned staged row");
    auto group_sync = [&]() { asm volatile("bar.sync 1, %0;" : : "r"(WT) : "memory"); };
    const int n = sh_comp_ndof[c];
    const int sb = sh_comp_begin[c];
    const int nrow = sh_comp_nrow[c];
    const int rb = sh_comp_row_begin[c];
    const int wbeg = sh_comp_wbegin[c];
    const int np = n * (n + 1) / 2;
    const bool own_row = lane < n;                              // warp 0: lane = DOF
    const int rowp = lane * (lane + 1) / 2;
    const bool resident = nrow <= WCH;
    float* const Hpk = sh_arena;                                // packed M_c / H / L of the component
    float* const pool = sh_arena + MPW;                         // staged rows
    float* const qc = sh_q + sb;                                // iterate in the world slots
    // the thread's packed entries E = tid + WT k, k < kmax (E < np); (a, b) is decoded from E where
    // needed (no loop-carried table: registers are the wide path's scarce resource) and the M_c
    // values go to the world's scratch once (the same thread reads them back every round)
    const int kmax = (np - tid + WT - 1) / WT;
    auto packed_ab = [&](const int E, int& a, int& b) {
      a = (int)((sqrtf(8.0f * (float)E + 1.0f) - 1.0f) * 0.5f);
      while ((a + 1) * (a + 2) / 2 <= E) ++a;
      while (a * (a + 1) / 2 > E) --a;
      b = E - a * (a + 1) / 2;
    };
#pragma unroll
    for (int k = 0; k < EPT; ++k) {
      if (k >= kmax) break;
      const int E = tid + WT * k;
      int a, b;
      packed_ab(E, a, b);
      const int ga_ = sh_slot_dof[sb + a];
      const int gb = sh_slot_dof[sb + b];
      const int elemid = M_elemid_p[max(ga_, gb) * s_M_elemid + min(ga_, gb)];
      wideM_p[E] = (elemid >= 0) ? M_p[elemid] : 0.0f;
    }
    float qfs = 0.0f;
    if (warp == 0 && own_row) {
      const int g = sh_slot_dof[sb + lane];
      qfs = qfrc_smooth_p[g];
      qc[lane] = warmstart ? warm_p[g] : qacc_smooth_p[g];
    }
    if (tid == 0) sh_flags[4] = 0;  // warp 0's round decision for the group
    // classify row r (record w) at the current iterate: coef = D on QUADRATIC rows, F = -D x;
    // force/state are published every round (the final round wins)
    auto classify_row = [&](const int r, float4* const w) {
      float x = -w[WR4 - 1].y;
#pragma unroll
      for (int b = 0; b < MN; ++b) {
        if (b >= n) break;
        const float4 wb = w[b >> 2];
        const float vb = ((b & 3) == 0) ? wb.x : (((b & 3) == 1) ? wb.y : (((b & 3) == 2) ? wb.z : wb.w));
        x += vb * qc[b];
      }
      const bool quad = (r < ne) || (x < 0.0f);
      const float Dv = w[WR4 - 1].x;
      const float f = quad ? -Dv * x : 0.0f;
      w[WR4 - 1].z = quad ? Dv : 0.0f;
      w[WR4 - 1].w = f;
      force_o[r] = f;
      state_o[r] = quad ? STATE_QUADRATIC : STATE_SATISFIED;
    };
    group_sync();
    float gd = 0.0f;
    float sv = 0.0f;
    int status = STATUS_CERTIFIED;
    int rounds = 0;
    float ga = 0.0f;
    for (int round = 0; round < round_cap; ++round) {
      // packed entries start from M_c (scratch loads in flight with the first row loads)
      float hp[EPT];
#pragma unroll
      for (int k = 0; k < EPT; ++k) {
        hp[k] = 0.0f;
        if (k < kmax) hp[k] = wideM_p[tid + WT * k];
      }
      ga = 0.0f;
      float mq = 0.0f;
      const int trow = own_row ? lane : 0;
      for (int cb = 0; cb < nrow; cb += WCH) {
        const int cnt = min(WCH, nrow - cb);
        // thread t stages row cb + t (resident rows after the first round: reclassify in place)
        if (tid < cnt && !DBG_NO_STAGE) {
          const int r = sh_row_order[rb + cb + tid];
          float4* const row4 = reinterpret_cast<float4*>(pool + tid * TS);
          float4 w[WR4];
          if (resident && round > 0) {
#pragma unroll
            for (int i = 0; i < WR4; ++i) w[i] = row4[i];
            classify_row(r, w);
            row4[WR4 - 1] = w[WR4 - 1];
          } else {
            const int rec = wbeg + cb + tid;
#pragma unroll
            for (int i = 0; i < WR4; ++i) w[i] = wide4_p[WR4 * rec + i];
            classify_row(r, w);
#pragma unroll
            for (int i = 0; i < WR4; ++i) row4[i] = w[i];
          }
        }
        if (cb == 0) {
          // M_c into the arena for warp 0's M_c q (read before any thread publishes H)
#pragma unroll
          for (int k = 0; k < EPT; ++k) {
            if (k < kmax) Hpk[tid + WT * k] = hp[k];
          }
        }
        group_sync();
        if (cb == 0 && warp == 0 && own_row) {
#pragma unroll 4
          for (int b = 0; b < n; ++b) {
            const int hi = max(lane, b);
            const int lo = min(lane, b);
            mq += Hpk[hi * (hi + 1) / 2 + lo] * qc[b];
          }
        }
        // every group thread sweeps the staged rows for its live packed entries, one entry at a
        // time (the loads pipeline across rows; SATISFIED rows carry coef = 0); warp 0 lane a
        // adds F j_a
#pragma unroll
        for (int k = 0; k < EPT; ++k) {
          if (k >= kmax) break;
          int ea, eb;
          packed_ab(tid + WT * k, ea, eb);
          float h = hp[k];
#pragma unroll 4
          for (int t = 0; t < (DBG_NO_ACC ? 0 : cnt); ++t) {
            const float* const row = pool + t * TS;
            h += row[MN + 2] * row[ea] * row[eb];
          }
          hp[k] = h;
        }
        if (warp == 0) {
#pragma unroll 4
          for (int t = 0; t < (DBG_NO_ACC ? 0 : cnt); ++t) {
            const float* const row = pool + t * TS;
            ga += row[MN + 3] * row[trow];
          }
        }
        // the pool is restaged next pass; a single pass orders warp 0's M_c q before the H publish
        if (cb + WCH < nrow || resident) group_sync();
      }
      // H = M_c + J^T D J as packed entries
#pragma unroll
      for (int k = 0; k < EPT; ++k) {
        if (k < kmax) Hpk[tid + WT * k] = hp[k];
      }
      group_sync();
      if (warp == 0) {
        const float gv = own_row ? (mq - qfs - ga) : 0.0f;
        // lane a owns row a of H in registers (zero beyond the row and beyond n)
        float Hrow[MN];
#pragma unroll
        for (int b = 0; b < MN; ++b) Hrow[b] = (own_row && b <= lane) ? Hpk[rowp + b] : 0.0f;
        // packed Cholesky, lane = row, right-looking; inv_own = 1/L[lane][lane]. The column
        // shuffles run in chunks of CH straight-line steps (pipelined, bounded register pressure);
        // every loop has a constant trip count so Hrow stays in registers
        bool ok = true;
        float inv_own = 0.0f;
#pragma unroll
        for (int p = 0; p < MN; ++p) {
          if (p >= n || DBG_NO_FACTOR) break;
          const float d = __shfl_sync(FULL, Hrow[p], p);
          ok = ok && (d > 0.0f) && isfinite(d);
          const float inv = rsqrtf(d);
          if (lane == p) inv_own = inv;
          if (lane > p) Hrow[p] *= inv;  // L[lane][p]
          const float lp = Hrow[p];
#pragma unroll
          for (int jb = 0; jb < MN; jb += CH) {
            if (jb + CH <= p + 1) continue;  // chunk at or before the pivot (folded)
            if (jb >= n) break;
#pragma unroll
            for (int j = jb; j < jb + CH && j < MN; ++j) {
              if (j > p) {
                const float ljp = __shfl_sync(FULL, lp, j);
                if (j < n && lane >= j) Hrow[j] -= lp * ljp;
              }
            }
          }
        }
        rounds = round + 1;
        int stop = 0;
        if (!ok) {
          status = STATUS_NOT_PD;
          stop = 1;
        } else {
          gd = gv * gv;
#pragma unroll
          for (int o = 16; o > 0; o >>= 1) gd += __shfl_xor_sync(FULL, gd, o);
          // forward: y = L^-1 grad (lane k publishes y_k at step k)
          float acc = gv;
          float y_own = 0.0f;
#pragma unroll
          for (int k = 0; k < MN; ++k) {
            if (k >= n || DBG_NO_FACTOR) break;
            const float yk = __shfl_sync(FULL, acc * inv_own, k);
            if (lane == k) y_own = yk;
            if (lane > k) acc -= Hrow[k] * yk;
          }
          // packed L to the arena (own row) for the backward solve
#pragma unroll
          for (int b = 0; b < MN; ++b) {
            if (own_row && b <= lane) Hpk[rowp + b] = Hrow[b];
          }
          __syncwarp();
          // backward: s = L^-T y (L[j][lane] from the arena, loaded CH steps ahead of the chain)
          acc = y_own;
          float s_own = 0.0f;
#pragma unroll
          for (int jb = MN - CH; jb >= 0; jb -= CH) {
            if (jb >= n) continue;  // chunk beyond n
            float lj[CH];
#pragma unroll
            for (int i = 0; i < CH; ++i) {
              const int j = jb + i;
              lj[i] = (j < n && own_row && lane < j) ? Hpk[j * (j + 1) / 2 + lane] : 0.0f;
            }
#pragma unroll
            for (int i = CH - 1; i >= 0; --i) {
              const int j = jb + i;
              const bool live = (j < n) && !DBG_NO_FACTOR;
              const float sj = __shfl_sync(FULL, acc * inv_own, j);
              if (live && lane == j) s_own = sj;
              if (live && lane < j) acc -= lj[i] * sj;
            }
          }
          sv = s_own * gv;
#pragma unroll
          for (int o = 16; o > 0; o >>= 1) sv += __shfl_xor_sync(FULL, sv, o);
          const bool finite_ok = isfinite(gd) && isfinite(sv);
          const float grad_r = sqrtf(gd) / scale;
          const float mi_r = 0.5f * sv / scale;
          const bool pass = finite_ok && ((grad_r < comp_tol) || (mi_r < comp_tol));
          const bool last = (round + 1 >= round_cap);
          if (!finite_ok) {
            status = STATUS_NONFINITE;
            stop = 1;
          } else if (pass || last) {
            stop = 1;
          } else {
            // semismooth Newton step on the current active set
            __syncwarp();
            if (own_row) qc[lane] -= s_own;
          }
        }
        if (lane == 0) sh_flags[4] = stop;
      }
      group_sync();
      if (sh_flags[4]) break;
    }
    // force/state were stored by the last round's classification; the iterate is already in sh_q
    if (warp == 0) {
      if (own_row) sh_qfrc[sb + lane] = ga;
      if (lane == 0) {
        sh_comp_status[c] = (unsigned char)status;
        sh_comp_rounds[c] = (unsigned char)rounds;
        sh_comp_gd[c] = gd;
        sh_comp_dec[c] = sv;
      }
    }
    group_sync();
  };

  // ---------------------------------------------------------------- phase 1a: wide components (group)
  // warps 0..WG-1 solve the wide components in work order; the other warps start pulling below
  if (warp < WG) {
    for (int i = 0; i < ncomp_active; ++i) {
      const int c = sh_comp_order[i];
      if ((int)sh_comp_ndof[c] > MNS) wide_component(c);
    }
  }
  // ---------------------------------------------------------------- phase 1b: warps pull the rest
  for (;;) {
    int next = 0;
    if (lane == 0) next = atomicAdd(&sh_flags[3], 1);
    next = __shfl_sync(FULL, next, 0);
    if (next >= ncomp_active) break;
    const int c = sh_comp_order[next];
    const int nd = sh_comp_ndof[c];
    if (nd > MNS) continue;
    if (nd <= LN) light_component(c);
    else if (nd <= M9) mid_component((char(*)[M9])nullptr, c);
    else mid_component((char(*)[MNS])nullptr, c);
  }
  __syncthreads();

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
      max_rounds = max(max_rounds, (int)sh_comp_rounds[c]);
    }
    const float grad_r = sqrtf(world_gd) / scale;
    const float mi_r = 0.5f * world_dec / scale;
    const bool pass = isfinite(grad_r) && isfinite(mi_r) && ((grad_r < ctol) || (mi_r < ctol));
    if (status == STATUS_CERTIFIED && !pass) status = STATUS_UNCERTIFIED;
    status_o[worldid] = status;
    gradient_o[worldid] = grad_r;
    decrement_o[worldid] = mi_r;
    niter_o[worldid] = max_rounds;
    if (status != STATUS_CERTIFIED) {
      stock_o[worldid] = 1;
      atomicAdd(nstock_o, 1);
      atomicAdd(nstock_total_o, 1);
    } else {
      stock_o[worldid] = 0;
    }
  }
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
  // rows without active columns (output-only): Jaref = -aref
  for (int r = tid; r < nefc; r += BLOCK) {
    if (sh_row_comp[r] != ROW_NONE) continue;
    const float x = -aref_p[r];
    const bool quad = (r < ne) || (x < 0.0f);
    force_o[r] = quad ? -D_p[r] * x : 0.0f;
    state_o[r] = quad ? STATE_QUADRATIC : STATE_SATISFIED;
  }
}
"""

_RANK_SNIPPET = r"""
{
  const unsigned FULL = 0xffffffffu;
  const int* const nefc_p = reinterpret_cast<const int*>(nefc.data);
  int* const order_p = reinterpret_cast<int*>(world_order.data);
  const int nworld = (int)nefc.shape[0];
  const int key = nefc_p[worldid];
  int cnt = 0;
  for (int v = lane; v < nworld; v += 32) {
    const int kv = nefc_p[v];
    if (kv > key || (kv == key && v < worldid)) ++cnt;
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) cnt += __shfl_xor_sync(FULL, cnt, o);
  if (lane == 0) order_p[cnt] = worldid;
}
"""


def _render(nv: int, njmax: int, debug_exit: int = 0, round_cap: int = ROUND_CAP) -> str:
  replacements = {
    "__BLOCK__": str(BLOCK_DIM),
    "__ROW_CAP__": str(ROW_CAP),
    "__LN__": str(LIGHT_DOF_CAP),
    "__RES__": str(RESIDENT_ROWS_PER_LANE),
    "__DR4__": str(DENSE_ROW_FLOATS // 4),
    "__WIDE_CAP__": str(WIDE_ROW_CAP),
    "__WR4__": str(WIDE_ROW_FLOATS // 4),
    "__M9__": str(REGISTER_FACTOR_DOF_CAP),
    "__WG__": str(WIDE_GROUP_WARPS),
    "__MN__": str(COMP_DOF_CAP),
    "__NCDOF__": str(NCDOF_CAP),
    "__NTREE__": str(NTREE_CAP),
    "__NV__": str(nv),
    "__ROUND_CAP__": str(round_cap),
    "__DEBUG_EXIT__": str(debug_exit),
    "__STATE_SATISFIED__": str(int(types.ConstraintState.SATISFIED)),
    "__STATE_QUADRATIC__": str(int(types.ConstraintState.QUADRATIC)),
  }
  source = _NATIVE_TEMPLATE
  for marker, value in replacements.items():
    source = source.replace(marker, value)
  return source


_KERNELS: dict[tuple[int, int, int, int], wp.Kernel] = {}


def world_solver_kernel(nv: int, njmax: int, debug_exit: int = 0, round_cap: int = ROUND_CAP) -> wp.Kernel:
  """Build (and cache) the per-world solver kernel for a model size and round cap."""
  key = (nv, njmax, debug_exit, round_cap)
  kernel = _KERNELS.get(key)
  if kernel is not None:
    return kernel

  @wp.func_native(snippet=_render(nv, njmax, debug_exit, round_cap))
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
    dense_rows: wp.array3d(dtype=float),
    wide_rows: wp.array3d(dtype=float),
    wide_M: wp.array2d(dtype=float),
    qacc_out: wp.array2d(dtype=float),
    qfrc_constraint_out: wp.array2d(dtype=float),
    efc_force_out: wp.array2d(dtype=float),
    efc_state_out: wp.array2d(dtype=int),
    efc_Ma_out: wp.array2d(dtype=float),
    solver_niter_out: wp.array(dtype=int),
    stock_world_out: wp.array(dtype=int),
    nstock_out: wp.array(dtype=int),
    nstock_total_out: wp.array(dtype=int),
    world_status_out: wp.array(dtype=int),
    world_gradient_out: wp.array(dtype=float),
    world_decrement_out: wp.array(dtype=float),
  ): ...

  @wp.kernel(module="unique", enable_backward=False, grid_stride=False, launch_bounds=(BLOCK_DIM, MIN_BLOCKS_PER_SM))
  def kernel(
    ntree: int,
    njmax: int,
    nv_scale: int,
    warmstart: int,
    world_order: wp.array(dtype=int),
    M_elemid: wp.array2d(dtype=int),
    M_mulm_rowadr: wp.array(dtype=int),
    M_mulm_col: wp.array(dtype=int),
    M_mulm_madr: wp.array(dtype=int),
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
    dense_rows: wp.array3d(dtype=float),
    wide_rows: wp.array3d(dtype=float),
    wide_M: wp.array2d(dtype=float),
    qacc_out: wp.array2d(dtype=float),
    qfrc_constraint_out: wp.array2d(dtype=float),
    efc_force_out: wp.array2d(dtype=float),
    efc_state_out: wp.array2d(dtype=int),
    efc_Ma_out: wp.array2d(dtype=float),
    solver_niter_out: wp.array(dtype=int),
    stock_world_out: wp.array(dtype=int),
    nstock_out: wp.array(dtype=int),
    nstock_total_out: wp.array(dtype=int),
    world_status_out: wp.array(dtype=int),
    world_gradient_out: wp.array(dtype=float),
    world_decrement_out: wp.array(dtype=float),
  ):
    block, tid = wp.tid()
    worldid = block
    if world_order.shape[0] > 0:
      worldid = world_order[block]
      if worldid < 0:
        # padded entry of a flagged-world list (uniform per CTA: no thread reaches a barrier)
        return
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
      dense_rows,
      wide_rows,
      wide_M,
      qacc_out,
      qfrc_constraint_out,
      efc_force_out,
      efc_state_out,
      efc_Ma_out,
      solver_niter_out,
      stock_world_out,
      nstock_out,
      nstock_total_out,
      world_status_out,
      world_gradient_out,
      world_decrement_out,
    )

  _KERNELS[key] = kernel
  return kernel


@wp.func_native(snippet=_RANK_SNIPPET)
def _rank_world(worldid: int, lane: int, nefc: wp.array(dtype=int), world_order: wp.array(dtype=int)): ...


@wp.kernel(module="unique", enable_backward=False, grid_stride=False)
def _rank_worlds_by_rows(nefc: wp.array(dtype=int), world_order: wp.array(dtype=int)):
  """Stable descending sort of worlds by row count (one warp per world)."""
  worldid, lane = wp.tid()
  _rank_world(worldid, lane, nefc, world_order)


@dataclasses.dataclass
class WorldSolverContext:
  """Per-world routing and certificate telemetry of the world solver.

  Attributes:
    stock_world: 1 if the world must take the stock solve, else 0    (nworld,)
    nstock: number of worlds routed to the stock solve                (1,)
    nstock_total: stock-routed worlds accumulated over launches (when the
      context is created under graph capture its zero-init is replayed too, so
      it counts per replay)                                            (1,)
    status: STATUS_* code per world                                   (nworld,)
    gradient: rescaled gradient norm of the world certificate         (nworld,)
    decrement: rescaled half Newton decrement of the certificate      (nworld,)
    world_order: CTA -> world permutation (heavy worlds first)        (nworld,)
    flagged: worlds flagged by the last launch, padded with -1         (RESOLVE_WORLD_CAP,)
    flagged_count: number of flagged worlds appended to ``flagged``    (1,)
    dense_rows: dense row records (j[0..16), D, aref)                  (nworld, ROW_CAP, 20)
    wide_rows: dense records of rows of 17..32 DOF components           (nworld, WIDE_ROW_CAP, 36)
    wide_M: packed M_c of the 17..32 DOF component being solved         (nworld, 528)
  """

  stock_world: wp.array
  nstock: wp.array
  nstock_total: wp.array
  status: wp.array
  gradient: wp.array
  decrement: wp.array
  world_order: wp.array
  flagged: wp.array
  flagged_count: wp.array
  dense_rows: wp.array
  wide_rows: wp.array
  wide_M: wp.array


def create_world_solver_context(nworld: int, device=None) -> WorldSolverContext:
  return WorldSolverContext(
    stock_world=wp.zeros(nworld, dtype=int, device=device),
    nstock=wp.zeros(1, dtype=int, device=device),
    nstock_total=wp.zeros(1, dtype=int, device=device),
    status=wp.zeros(nworld, dtype=int, device=device),
    gradient=wp.zeros(nworld, dtype=float, device=device),
    decrement=wp.zeros(nworld, dtype=float, device=device),
    world_order=wp.zeros(nworld, dtype=int, device=device),
    flagged=wp.full(RESOLVE_WORLD_CAP, -1, dtype=int, device=device),
    flagged_count=wp.zeros(1, dtype=int, device=device),
    dense_rows=wp.empty((nworld, ROW_CAP, DENSE_ROW_FLOATS), dtype=float, device=device),
    wide_rows=wp.empty((nworld, WIDE_ROW_CAP, WIDE_ROW_FLOATS), dtype=float, device=device),
    wide_M=wp.empty((nworld, COMP_DOF_CAP * (COMP_DOF_CAP + 1) // 2), dtype=float, device=device),
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
  dof_treeid=None,
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
  heavy_first: bool = True,
  order: wp.array | None = None,
  round_cap: int = ROUND_CAP,
):
  """Launch the per-world solver on explicit arrays (zeroes ``ctx.nstock`` first).

  ``dof_treeid`` is accepted for call-site compatibility and unused.
  """
  del dof_treeid
  if order is None:
    ctx.nstock.zero_()
    if heavy_first:
      wp.launch_tiled(_rank_worlds_by_rows, dim=nworld, inputs=[nefc], outputs=[ctx.world_order], block_dim=32)
      order = ctx.world_order
  wp.launch_tiled(
    world_solver_kernel(nv, njmax, debug_exit, round_cap),
    dim=nworld,
    inputs=[
      ntree,
      njmax,
      nv_scale,
      int(warmstart),
      order,
      M_elemid,
      M_mulm_rowadr,
      M_mulm_col,
      M_mulm_madr,
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
      ctx.dense_rows,
      ctx.wide_rows,
      ctx.wide_M,
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
      ctx.nstock_total,
      ctx.status,
      ctx.gradient,
      ctx.decrement,
    ],
    block_dim=BLOCK_DIM,
  )


@wp.kernel(module="unique", enable_backward=False)
def _collect_flagged_worlds(
  stock_world: wp.array(dtype=int),
  flagged: wp.array(dtype=int),
  flagged_count: wp.array(dtype=int),
  nstock: wp.array(dtype=int),
):
  """Append flagged worlds to the padded list; worlds past its capacity stay counted in nstock."""
  worldid = wp.tid()
  if stock_world[worldid] == 0:
    return
  slot = wp.atomic_add(flagged_count, 0, 1)
  if slot < flagged.shape[0]:
    flagged[slot] = worldid
  else:
    wp.atomic_add(nstock, 0, 1)


def world_resolve_flagged(m: types.Model, d: types.Data, ctx: WorldSolverContext):
  """Re-run the world solver with RESOLVE_ROUND_CAP rounds on the worlds the last launch flagged.

  ``ctx.nstock`` is rebuilt: worlds that certify now clear their flag, the others (and any beyond
  RESOLVE_WORLD_CAP) stay routed to the stock solve. A flagged first pass leaves its inputs
  untouched, so the re-solve sees the same problem.
  """
  ctx.nstock.zero_()
  ctx.flagged_count.zero_()
  ctx.flagged.fill_(-1)
  wp.launch(_collect_flagged_worlds, dim=d.nworld, inputs=[ctx.stock_world, ctx.flagged, ctx.flagged_count, ctx.nstock])
  _launch_world_solver_impl(m, d, ctx, order=ctx.flagged, nworld=RESOLVE_WORLD_CAP, round_cap=RESOLVE_ROUND_CAP)


def world_solve(m: types.Model, d: types.Data, ctx: WorldSolverContext, heavy_first: bool = True):
  """Run the per-world solver on canonical Data arrays."""
  _launch_world_solver_impl(m, d, ctx, heavy_first=heavy_first)


def _launch_world_solver_impl(
  m: types.Model,
  d: types.Data,
  ctx: WorldSolverContext,
  heavy_first: bool = True,
  order: wp.array | None = None,
  nworld: int | None = None,
  round_cap: int = ROUND_CAP,
):
  launch_world_solver(
    nworld=d.nworld if nworld is None else nworld,
    nv=m.nv,
    ntree=m.ntree,
    njmax=d.njmax,
    nv_scale=d.nvmax_pad,
    warmstart=not (m.opt.disableflags & types.DisableBit.WARMSTART),
    M_elemid=m.M_elemid,
    M_mulm_rowadr=m.M_mulm_rowadr,
    M_mulm_col=m.M_mulm_col,
    M_mulm_madr=m.M_mulm_madr,
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
    heavy_first=heavy_first,
    order=order,
    round_cap=round_cap,
  )
