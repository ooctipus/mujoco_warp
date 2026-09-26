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

import warp as wp

from mujoco_warp._src import types
from mujoco_warp._src.types import EqType
from mujoco_warp._src.types import ObjType
from mujoco_warp._src.types import SleepPolicy
from mujoco_warp._src.types import SleepState
from mujoco_warp._src.types import WrapType
from mujoco_warp._src.warp_util import event_scope

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

# tree_asleep value for fully awake tree
K_AWAKE_VAL = -(1 + types.MJ_MINAWAKE)


@wp.func
def _sleep_cycle(tree_asleep: wp.array2d[int], ntree: int, worldid: int, treeid: int) -> int:  # kernel_analyzer: ignore
  if treeid < 0 or treeid >= ntree:
    return -1

  smallest = int(treeid)
  current = int(treeid)
  count = int(0)

  for step in range(ntree + 1):  # safe upper bound
    next_tree = tree_asleep[worldid, current]
    if next_tree < 0 or next_tree >= ntree:
      return -1

    if next_tree < smallest:
      smallest = next_tree

    current = next_tree
    count += 1
    if current == treeid:
      break

  return smallest


@wp.func
def _tendon_limit_active(
  tendon_limited: wp.array[int],  # kernel_analyzer: ignore
  tendon_range: wp.array2d[wp.vec2],  # kernel_analyzer: ignore
  tendon_margin: wp.array2d[float],  # kernel_analyzer: ignore
  ten_length: wp.array2d[float],  # kernel_analyzer: ignore
  worldid: int,  # kernel_analyzer: ignore
  tenid: int,
) -> bool:
  if tendon_limited[tenid] == 0:
    return False

  length = ten_length[worldid, tenid]
  margin = tendon_margin[worldid % tendon_margin.shape[0], tenid]
  r = tendon_range[worldid % tendon_range.shape[0], tenid]

  # low limit
  dist_low = length - r[0]
  if dist_low < margin:
    return True

  # high limit
  dist_high = r[1] - length
  if dist_high < margin:
    return True

  return False


@wp.kernel
def _zero_sleep_counters(
  # In:
  world_mask: wp.array[bool],
  # Data out:
  ntree_awake_out: wp.array[int],
  nbody_awake_out: wp.array[int],
  nv_awake_out: wp.array[int],
):
  worldid = wp.tid()
  if world_mask and not world_mask[worldid]:
    return
  ntree_awake_out[worldid] = 0
  nbody_awake_out[worldid] = 0
  nv_awake_out[worldid] = 0


@wp.kernel
def _update_sleep_trees(
  # Model:
  tree_sleep_policy: wp.array2d[int],
  # Data in:
  tree_asleep_in: wp.array2d[int],
  # In:
  world_mask: wp.array[bool],
  # Data out:
  ntree_awake_out: wp.array[int],
  tree_awake_out: wp.array2d[int],
):
  worldid, treeid = wp.tid()
  if world_mask and not world_mask[worldid]:
    return
  if tree_sleep_policy[worldid % tree_sleep_policy.shape[0], treeid] == SleepPolicy.ALWAYS:
    tree_awake_out[worldid, treeid] = 0
    return

  is_awake = int(tree_asleep_in[worldid, treeid] < 0)
  tree_awake_out[worldid, treeid] = is_awake
  if is_awake == 1:
    wp.atomic_add(ntree_awake_out, worldid, 1)


@wp.kernel
def _update_sleep_bodies(
  # Model:
  body_parentid: wp.array[int],
  body_rootid: wp.array[int],
  body_mocapid: wp.array[int],
  body_treeid: wp.array[int],
  tree_sleep_policy: wp.array2d[int],
  # Data in:
  tree_awake_in: wp.array2d[int],
  # In:
  flg_staticawake: int,
  compact: bool,
  world_mask: wp.array[bool],
  # Data out:
  nbody_awake_out: wp.array[int],
  body_awake_out: wp.array2d[int],
  body_awake_ind_out: wp.array2d[int],
):
  worldid, bodyid = wp.tid()
  if world_mask and not world_mask[worldid]:
    return

  tree = body_treeid[bodyid]

  # check static
  if tree < 0:
    root = body_rootid[bodyid]
    mocap = body_mocapid[root]
    if mocap >= 0:
      state = SleepState.AWAKE
    else:
      state = SleepState.AWAKE if flg_staticawake != 0 else SleepState.STATIC
  else:
    policy = tree_sleep_policy[worldid % tree_sleep_policy.shape[0], tree]
    if policy == SleepPolicy.ALWAYS:
      state = SleepState.ASLEEP
    else:
      state = SleepState.AWAKE if tree_awake_in[worldid, tree] == 1 else SleepState.ASLEEP

  body_awake_out[worldid, bodyid] = state

  if state != SleepState.ASLEEP:
    idx = wp.atomic_add(nbody_awake_out, worldid, 1)
    if not compact:
      idx = bodyid
    body_awake_ind_out[worldid, idx] = bodyid


@wp.kernel
def _update_sleep_dofs(
  # Model:
  body_treeid: wp.array[int],
  dof_bodyid: wp.array[int],
  # Data in:
  body_awake_in: wp.array2d[int],
  # In:
  compact: bool,
  world_mask: wp.array[bool],
  # Data out:
  nv_awake_out: wp.array[int],
  dof_awake_ind_out: wp.array2d[int],
):
  worldid, dofid = wp.tid()
  if world_mask and not world_mask[worldid]:
    return
  bodyid = dof_bodyid[dofid]
  if body_treeid[bodyid] >= 0 and body_awake_in[worldid, bodyid] == SleepState.AWAKE:
    idx = wp.atomic_add(nv_awake_out, worldid, 1)
    if not compact:
      idx = dofid
    dof_awake_ind_out[worldid, idx] = dofid


@event_scope
def update_sleep(m: types.Model, d: types.Data, flg_staticawake: int = 0, *, world_mask: wp.array | None = None):
  """Computes sleeping arrays from tree_asleep."""
  compact = bool(m.opt.enableflags & types.EnableBit.SLEEP) and not bool(m.opt.disableflags & types.DisableBit.ISLAND)
  wp.launch(
    _zero_sleep_counters,
    dim=d.nworld,
    inputs=[world_mask],
    outputs=[d.ntree_awake, d.nbody_awake, d.nv_awake],
  )

  wp.launch(
    _update_sleep_trees,
    dim=(d.nworld, m.ntree),
    inputs=[m.tree_sleep_policy, d.tree_asleep, world_mask],
    outputs=[d.ntree_awake, d.tree_awake],
  )

  wp.launch(
    _update_sleep_bodies,
    dim=(d.nworld, m.nbody),
    inputs=[
      m.body_parentid,
      m.body_rootid,
      m.body_mocapid,
      m.body_treeid,
      m.tree_sleep_policy,
      d.tree_awake,
      flg_staticawake,
      compact,
      world_mask,
    ],
    outputs=[
      d.nbody_awake,
      d.body_awake,
      d.body_awake_ind,
    ],
  )

  wp.launch(
    _update_sleep_dofs,
    dim=(d.nworld, m.nv),
    inputs=[
      m.body_treeid,
      m.dof_bodyid,
      d.body_awake,
      compact,
      world_mask,
    ],
    outputs=[d.nv_awake, d.dof_awake_ind],
  )


@event_scope
def update_sleep_trees(m: types.Model, d: types.Data):
  """Lightweight update of tree_awake array only, avoiding body/dof kernel launches."""
  wp.launch(
    _zero_sleep_counters,
    dim=d.nworld,
    inputs=[None],
    outputs=[d.ntree_awake, d.nbody_awake, d.nv_awake],
  )

  wp.launch(
    _update_sleep_trees,
    dim=(d.nworld, m.ntree),
    inputs=[m.tree_sleep_policy, d.tree_asleep, None],
    outputs=[d.ntree_awake, d.tree_awake],
  )


@wp.func
def _wake_tree(
  # Model:
  ntree: int,
  tree_sleep_policy: wp.array2d[int],
  # In:
  worldid: int,
  treeid: int,
  wakeval: int,
  # Data out:
  tree_asleep_out: wp.array2d[int],
) -> int:
  """Wakes tree treeid and its associated cycle, returning number of woke trees."""
  if treeid < 0 or treeid >= ntree:
    return 0

  w_policy = worldid % tree_sleep_policy.shape[0]
  if tree_sleep_policy[w_policy, treeid] == SleepPolicy.ALWAYS:
    return 0

  asleep_val = tree_asleep_out[worldid, treeid]
  if asleep_val < 0:
    if wakeval < asleep_val:
      tree_asleep_out[worldid, treeid] = wakeval
    return 0

  nwoke = int(0)
  current = int(treeid)
  for step in range(ntree + 1):  # safe upper bound
    next_tree = tree_asleep_out[worldid, current]
    if next_tree < 0 or next_tree >= ntree:
      break

    if tree_sleep_policy[w_policy, current] != SleepPolicy.ALWAYS:
      tree_asleep_out[worldid, current] = wakeval
      nwoke += 1
    current = next_tree
    if current == treeid:
      break

  return nwoke


@wp.func
def _tree_can_sleep(
  # Model:
  nbody: int,
  body_treeid: wp.array[int],
  dof_length: wp.array[float],
  tree_dofadr: wp.array[int],
  tree_dofnum: wp.array[int],
  tree_sleep_policy: wp.array2d[int],
  # Data in:
  qvel_in: wp.array2d[float],
  qfrc_applied_in: wp.array2d[float],
  xfrc_applied_in: wp.array2d[wp.spatial_vector],
  # In:
  worldid: int,
  treeid: int,
  sleep_tolerance: float,
) -> bool:
  policy = tree_sleep_policy[worldid % tree_sleep_policy.shape[0], treeid]
  if policy == SleepPolicy.AUTO_NEVER or policy == SleepPolicy.ALWAYS:
    return False

  # check xfrc_applied
  for b in range(nbody):
    if body_treeid[b] == treeid:
      xfrc = xfrc_applied_in[worldid, b]
      for i in range(6):
        if xfrc[i] != 0.0:
          return False

  # check qfrc_applied
  dofadr = tree_dofadr[treeid]
  dofnum = tree_dofnum[treeid]
  for d in range(dofnum):
    if qfrc_applied_in[worldid, dofadr + d] != 0.0:
      return False

  # check qvel
  for d in range(dofnum):
    dof_idx = dofadr + d
    v = qvel_in[worldid, dof_idx]
    weight = dof_length[dof_idx]
    if sleep_tolerance > 0.0:
      if wp.abs(weight * v) >= sleep_tolerance:
        return False
    else:
      if v != 0.0:
        return False

  return True


@wp.kernel
def _wake_kernel(
  # Model:
  nbody: int,
  ntree: int,
  body_treeid: wp.array[int],
  dof_length: wp.array[float],
  tree_dofadr: wp.array[int],
  tree_dofnum: wp.array[int],
  tree_sleep_policy: wp.array2d[int],
  # Data in:
  qvel_in: wp.array2d[float],
  qfrc_applied_in: wp.array2d[float],
  xfrc_applied_in: wp.array2d[wp.spatial_vector],
  tree_awake_in: wp.array2d[int],
  # Out:
  tree_asleep_out: wp.array2d[int],  # kernel_analyzer: ignore
):
  worldid, treeid = wp.tid()

  if tree_sleep_policy[worldid % tree_sleep_policy.shape[0], treeid] == SleepPolicy.ALWAYS:
    return

  asleep = int(tree_asleep_out[worldid, treeid] >= 0)
  if asleep == 0:
    return

  # if tree_awake mismatch or cannot sleep: wake up
  if tree_awake_in[worldid, treeid] == 1 or not _tree_can_sleep(
    nbody,
    body_treeid,
    dof_length,
    tree_dofadr,
    tree_dofnum,
    tree_sleep_policy,
    qvel_in,
    qfrc_applied_in,
    xfrc_applied_in,
    worldid,
    treeid,
    0.0,  # zero tolerance
  ):
    _wake_tree(ntree, tree_sleep_policy, worldid, treeid, K_AWAKE_VAL, tree_asleep_out)


@wp.kernel
def _wake_collision_kernel(
  # Model:
  ntree: int,
  body_treeid: wp.array[int],
  tree_sleep_policy: wp.array2d[int],
  geom_bodyid: wp.array[int],
  # Data in:
  tree_awake_in: wp.array2d[int],
  contact_geom_in: wp.array[wp.vec2i],
  contact_worldid_in: wp.array[int],
  nacon_in: wp.array[int],
  # Out:
  tree_asleep_out: wp.array2d[int],  # kernel_analyzer: ignore
):
  conid = wp.tid()
  if conid >= nacon_in[0]:
    return

  geom_pair = contact_geom_in[conid]
  g1 = geom_pair[0]
  g2 = geom_pair[1]

  if g1 < 0 or g2 < 0:
    return

  b1 = geom_bodyid[g1]
  b2 = geom_bodyid[g2]
  tree1 = body_treeid[b1]
  tree2 = body_treeid[b2]

  if tree1 < 0 or tree2 < 0:
    return

  worldid = contact_worldid_in[conid]
  awake1 = tree_awake_in[worldid, tree1]
  awake2 = tree_awake_in[worldid, tree2]

  if awake1 == 1 and awake2 == 1:
    return

  if awake1 == 0 and awake2 == 0:
    return

  # wake sleeping tree
  sleeping_tree = tree2 if awake1 == 1 else tree1
  if tree_sleep_policy[worldid % tree_sleep_policy.shape[0], sleeping_tree] == SleepPolicy.ALWAYS:
    return

  wakeval = tree_asleep_out[worldid, tree1] if awake1 == 1 else tree_asleep_out[worldid, tree2]

  _wake_tree(ntree, tree_sleep_policy, worldid, sleeping_tree, wakeval, tree_asleep_out)


@wp.kernel
def _wake_tendon_kernel(
  # Model:
  ntree: int,
  ntendon: int,
  body_treeid: wp.array[int],
  jnt_bodyid: wp.array[int],
  tree_sleep_policy: wp.array2d[int],
  geom_bodyid: wp.array[int],
  site_bodyid: wp.array[int],
  tendon_adr: wp.array[int],
  tendon_num: wp.array[int],
  tendon_limited: wp.array[int],
  tendon_range: wp.array2d[wp.vec2],
  tendon_margin: wp.array2d[float],
  wrap_type: wp.array[int],
  wrap_objid: wp.array[int],
  # Data in:
  ten_length_in: wp.array2d[float],
  tree_awake_in: wp.array2d[int],
  # Data out:
  tree_asleep_out: wp.array2d[int],
):
  worldid, tenid = wp.tid()

  adr = tendon_adr[tenid]
  num = tendon_num[tenid]

  # Pass 1: Check if any tree involved in the tendon is awake
  any_awake = int(0)
  wakeval = int(K_AWAKE_VAL)

  for i in range(num):
    idx = adr + i
    t_type = wrap_type[idx]
    t_objid = wrap_objid[idx]
    t = int(-1)
    if t_type == WrapType.JOINT:
      t = body_treeid[jnt_bodyid[t_objid]]
    elif t_type == WrapType.SITE:
      t = body_treeid[site_bodyid[t_objid]]
    elif t_type == WrapType.SPHERE or t_type == WrapType.CYLINDER:
      t = body_treeid[geom_bodyid[t_objid]]

    if t >= 0:
      if tree_awake_in[worldid, t] == 1:
        any_awake = 1
        val = tree_asleep_out[worldid, t]
        if val < wakeval:
          wakeval = val

  # Pass 2: If at least one tree is awake and the limit is active, wake up all sleeping trees
  if any_awake == 1:
    if _tendon_limit_active(tendon_limited, tendon_range, tendon_margin, ten_length_in, worldid, tenid):
      for i in range(num):
        idx = adr + i
        t_type = wrap_type[idx]
        t_objid = wrap_objid[idx]
        t = int(-1)
        if t_type == WrapType.JOINT:
          t = body_treeid[jnt_bodyid[t_objid]]
        elif t_type == WrapType.SITE:
          t = body_treeid[site_bodyid[t_objid]]
        elif t_type == WrapType.SPHERE or t_type == WrapType.CYLINDER:
          t = body_treeid[geom_bodyid[t_objid]]

        if t >= 0:
          if tree_awake_in[worldid, t] == 0:
            _wake_tree(ntree, tree_sleep_policy, worldid, t, wakeval, tree_asleep_out)


@wp.func
def _tendon_wake_val(
  # Model:
  body_treeid: wp.array[int],
  jnt_bodyid: wp.array[int],
  geom_bodyid: wp.array[int],
  site_bodyid: wp.array[int],
  tendon_adr: wp.array[int],
  tendon_num: wp.array[int],
  wrap_type: wp.array[int],
  wrap_objid: wp.array[int],
  # Data in:
  tree_awake_in: wp.array2d[int],
  # In:
  worldid: int,
  tenid: int,
  # Data out:
  tree_asleep_out: wp.array2d[int],
) -> int:
  """Returns the minimum (most-awake) tree_asleep value if any tree in tendon is awake, else 0."""
  if tenid < 0:
    return 0

  adr = tendon_adr[tenid]
  num = tendon_num[tenid]
  wakeval = int(0)

  for i in range(num):
    idx = adr + i
    t_type = wrap_type[idx]
    t_objid = wrap_objid[idx]
    t = int(-1)
    if t_type == WrapType.JOINT:
      t = body_treeid[jnt_bodyid[t_objid]]
    elif t_type == WrapType.SITE:
      t = body_treeid[site_bodyid[t_objid]]
    elif t_type == WrapType.SPHERE or t_type == WrapType.CYLINDER:
      t = body_treeid[geom_bodyid[t_objid]]

    if t >= 0:
      if tree_awake_in[worldid, t] == 1:
        val = tree_asleep_out[worldid, t]
        if wakeval == 0 or val < wakeval:
          wakeval = val

  return wakeval


@wp.func
def _wake_tendon_trees(
  # Model:
  ntree: int,
  body_treeid: wp.array[int],
  jnt_bodyid: wp.array[int],
  tree_sleep_policy: wp.array2d[int],
  geom_bodyid: wp.array[int],
  site_bodyid: wp.array[int],
  tendon_adr: wp.array[int],
  tendon_num: wp.array[int],
  wrap_type: wp.array[int],
  wrap_objid: wp.array[int],
  # Data in:
  tree_awake_in: wp.array2d[int],
  # In:
  worldid: int,
  tenid: int,
  wakeval: int,
  # Data out:
  tree_asleep_out: wp.array2d[int],
):
  """Wakes up all sleeping trees associated with a tendon."""
  if tenid < 0:
    return

  adr = tendon_adr[tenid]
  num = tendon_num[tenid]
  for i in range(num):
    idx = adr + i
    t_type = wrap_type[idx]
    t_objid = wrap_objid[idx]
    t = int(-1)
    if t_type == WrapType.JOINT:
      t = body_treeid[jnt_bodyid[t_objid]]
    elif t_type == WrapType.SITE:
      t = body_treeid[site_bodyid[t_objid]]
    elif t_type == WrapType.SPHERE or t_type == WrapType.CYLINDER:
      t = body_treeid[geom_bodyid[t_objid]]

    if t >= 0:
      if tree_awake_in[worldid, t] == 0:
        _wake_tree(ntree, tree_sleep_policy, worldid, t, wakeval, tree_asleep_out)


@wp.kernel
def _wake_equality_kernel(
  # Model:
  ntree: int,
  neq: int,
  body_treeid: wp.array[int],
  jnt_bodyid: wp.array[int],
  tree_sleep_policy: wp.array2d[int],
  geom_bodyid: wp.array[int],
  site_bodyid: wp.array[int],
  eq_type: wp.array[int],
  eq_obj1id: wp.array[int],
  eq_obj2id: wp.array[int],
  eq_objtype: wp.array[int],
  tendon_adr: wp.array[int],
  tendon_num: wp.array[int],
  wrap_type: wp.array[int],
  wrap_objid: wp.array[int],
  # Data in:
  eq_active_in: wp.array2d[bool],  # kernel_analyzer: ignore
  tree_awake_in: wp.array2d[int],
  # Data out:
  tree_asleep_out: wp.array2d[int],  # kernel_analyzer: ignore
):
  worldid, eqid = wp.tid()

  if not eq_active_in[worldid, eqid]:
    return

  eqtype = eq_type[eqid]
  id1 = eq_obj1id[eqid]
  id2 = eq_obj2id[eqid]

  if eqtype == EqType.CONNECT or eqtype == EqType.WELD or eqtype == EqType.JOINT:
    tree1 = int(-1)
    tree2 = int(-1)

    if eqtype == EqType.CONNECT or eqtype == EqType.WELD:
      if eq_objtype[eqid] == ObjType.BODY:
        tree1 = body_treeid[id1]
        tree2 = body_treeid[id2]
      else:
        tree1 = body_treeid[site_bodyid[id1]]
        tree2 = body_treeid[site_bodyid[id2]]
    elif eqtype == EqType.JOINT:
      tree1 = body_treeid[jnt_bodyid[id1]] if id1 >= 0 else -1
      tree2 = body_treeid[jnt_bodyid[id2]] if id2 >= 0 else -1

    w_policy = worldid % tree_sleep_policy.shape[0]
    if (tree1 >= 0 and tree_sleep_policy[w_policy, tree1] == SleepPolicy.ALWAYS) or (
      tree2 >= 0 and tree_sleep_policy[w_policy, tree2] == SleepPolicy.ALWAYS
    ):
      return

    s1 = tree_awake_in[worldid, tree1] if tree1 >= 0 else SleepState.STATIC
    s2 = tree_awake_in[worldid, tree2] if tree2 >= 0 else SleepState.STATIC

    if s1 != SleepState.ASLEEP and s2 != SleepState.ASLEEP:
      return
    if s1 == SleepState.STATIC or s2 == SleepState.STATIC:
      return
    if tree1 == tree2:
      return

    if s1 == SleepState.ASLEEP and s2 == SleepState.ASLEEP:
      cycle1 = _sleep_cycle(tree_asleep_out, ntree, worldid, tree1)
      cycle2 = _sleep_cycle(tree_asleep_out, ntree, worldid, tree2)
      if cycle1 != cycle2:
        _wake_tree(ntree, tree_sleep_policy, worldid, tree1, K_AWAKE_VAL, tree_asleep_out)
        _wake_tree(ntree, tree_sleep_policy, worldid, tree2, K_AWAKE_VAL, tree_asleep_out)
    else:
      sleeping_tree = tree1 if s1 == SleepState.ASLEEP else tree2
      _wake_tree(ntree, tree_sleep_policy, worldid, sleeping_tree, K_AWAKE_VAL, tree_asleep_out)

  elif eqtype == EqType.TENDON:
    ten1 = id1
    ten2 = id2
    w1 = _tendon_wake_val(
      body_treeid,
      jnt_bodyid,
      geom_bodyid,
      site_bodyid,
      tendon_adr,
      tendon_num,
      wrap_type,
      wrap_objid,
      tree_awake_in,
      worldid,
      ten1,
      tree_asleep_out,
    )
    w2 = _tendon_wake_val(
      body_treeid,
      jnt_bodyid,
      geom_bodyid,
      site_bodyid,
      tendon_adr,
      tendon_num,
      wrap_type,
      wrap_objid,
      tree_awake_in,
      worldid,
      ten2,
      tree_asleep_out,
    )

    if w1 < 0 or w2 < 0:
      wakeval = int(K_AWAKE_VAL)
      if w1 < 0 and w1 < wakeval:
        wakeval = w1
      if w2 < 0 and w2 < wakeval:
        wakeval = w2

      _wake_tendon_trees(
        ntree,
        body_treeid,
        jnt_bodyid,
        tree_sleep_policy,
        geom_bodyid,
        site_bodyid,
        tendon_adr,
        tendon_num,
        wrap_type,
        wrap_objid,
        tree_awake_in,
        worldid,
        ten1,
        wakeval,
        tree_asleep_out,
      )
      _wake_tendon_trees(
        ntree,
        body_treeid,
        jnt_bodyid,
        tree_sleep_policy,
        geom_bodyid,
        site_bodyid,
        tendon_adr,
        tendon_num,
        wrap_type,
        wrap_objid,
        tree_awake_in,
        worldid,
        ten2,
        wakeval,
        tree_asleep_out,
      )

  # TODO(team): Implement waking for EqType.FLEX constraints.


@event_scope
def wake(m: types.Model, d: types.Data):
  """Wakes sleeping trees due to user changes/perturbations."""
  wp.launch(
    _clear_disabled_dofs,
    dim=(d.nworld, m.nv),
    inputs=[m.dof_treeid, m.tree_sleep_policy, None, None],
    outputs=[d.qvel, d.qacc_warmstart, d.qfrc_applied, d.qacc],
  )
  wp.launch(
    _wake_kernel,
    dim=(d.nworld, m.ntree),
    inputs=[
      m.nbody,
      m.ntree,
      m.body_treeid,
      m.dof_length,
      m.tree_dofadr,
      m.tree_dofnum,
      m.tree_sleep_policy,
      d.qvel,
      d.qfrc_applied,
      d.xfrc_applied,
      d.tree_awake,
    ],
    outputs=[d.tree_asleep],
  )


@event_scope
def wake_collision(m: types.Model, d: types.Data):
  """Wakes sleeping trees that touch awake trees."""
  wp.launch(
    _wake_collision_kernel,
    dim=d.naconmax,
    inputs=[
      m.ntree,
      m.body_treeid,
      m.tree_sleep_policy,
      m.geom_bodyid,
      d.tree_awake,
      d.contact.geom,
      d.contact.worldid,
      d.nacon,
    ],
    outputs=[d.tree_asleep],
  )


@wp.kernel
def _wake_requested_trees(
  # Model:
  ntree: int,
  tree_sleep_policy: wp.array2d[int],
  # In:
  tree_mask: wp.array2d[int],
  world_mask: wp.array[bool],
  # Data out:
  tree_asleep_out: wp.array2d[int],
):
  worldid = wp.tid()
  if world_mask and not world_mask[worldid]:
    return
  # One walker per world preserves links when several requests share a sleeping island.
  for treeid in range(ntree):
    if not tree_mask or tree_mask[worldid, treeid] != 0:
      _wake_tree(ntree, tree_sleep_policy, worldid, treeid, K_AWAKE_VAL, tree_asleep_out)


def wake_trees(m: types.Model, d: types.Data, tree_mask: wp.array | None = None, *, world_mask: wp.array | None = None):
  """Wake requested trees and their sleeping islands, preserving permanent sleep policy.

  Args:
    m: Model owning the per-world sleep policies.
    d: Simulation data to update.
    tree_mask: Optional integer array of shape ``(nworld, ntree)``; nonzero requests a wake.
    world_mask: Optional boolean array of shape ``(nworld,)`` selecting worlds to modify.
  """
  if tree_mask is not None and (tree_mask.shape != (d.nworld, m.ntree) or tree_mask.dtype != wp.int32):
    raise ValueError("tree_mask must be an int32 array of shape (nworld, ntree).")
  if world_mask is not None and (world_mask.shape != (d.nworld,) or world_mask.dtype != wp.bool):
    raise ValueError("world_mask must be a boolean array of shape (nworld,).")
  wp.launch(
    _wake_requested_trees,
    dim=d.nworld,
    inputs=[m.ntree, m.tree_sleep_policy, tree_mask, world_mask],
    outputs=[d.tree_asleep],
  )
  update_sleep(m, d, world_mask=world_mask)


@wp.kernel
def _reset_sleep_trees(
  # Model:
  tree_sleep_policy: wp.array2d[int],
  # In:
  initial_tree_asleep: wp.array2d[int],
  world_mask: wp.array[bool],
  # Data out:
  tree_asleep_out: wp.array2d[int],
):
  worldid, treeid = wp.tid()
  if world_mask and not world_mask[worldid]:
    return
  value = K_AWAKE_VAL
  if initial_tree_asleep:
    w_initial = worldid % initial_tree_asleep.shape[0]
    value = initial_tree_asleep[w_initial, treeid]
    # Remove disabled members from the saved cycle without breaking links among enabled peers.
    for _ in range(tree_asleep_out.shape[1]):
      if value < 0:
        break
      policy = tree_sleep_policy[worldid % tree_sleep_policy.shape[0], value]
      if policy != SleepPolicy.ALWAYS and policy != SleepPolicy.AUTO_NEVER:
        break
      value = initial_tree_asleep[w_initial, value]
  if tree_sleep_policy[worldid % tree_sleep_policy.shape[0], treeid] == SleepPolicy.ALWAYS:
    value = treeid
  elif tree_sleep_policy[worldid % tree_sleep_policy.shape[0], treeid] == SleepPolicy.AUTO_NEVER:
    value = K_AWAKE_VAL
  tree_asleep_out[worldid, treeid] = value


@wp.kernel
def _clear_disabled_dofs(
  # Model:
  dof_treeid: wp.array[int],
  tree_sleep_policy: wp.array2d[int],
  # In:
  world_mask: wp.array[bool],
  tree_mask: wp.array2d[int],
  # Data out:
  qvel_out: wp.array2d[float],
  qacc_warmstart_out: wp.array2d[float],
  qfrc_applied_out: wp.array2d[float],
  qacc_out: wp.array2d[float],
):
  worldid, dofid = wp.tid()
  if world_mask and not world_mask[worldid]:
    return
  treeid = dof_treeid[dofid]
  if tree_mask and tree_mask[worldid, treeid] == 0:
    return
  if treeid >= 0 and tree_sleep_policy[worldid % tree_sleep_policy.shape[0], treeid] == SleepPolicy.ALWAYS:
    qvel_out[worldid, dofid] = 0.0
    qacc_out[worldid, dofid] = 0.0
    qacc_warmstart_out[worldid, dofid] = 0.0
    qfrc_applied_out[worldid, dofid] = 0.0


def reset_sleep(
  m: types.Model,
  d: types.Data,
  *,
  world_mask: wp.array | None = None,
  initial_tree_asleep: wp.array | None = None,
):
  """Reset sleep bookkeeping using current per-world policies without resetting coordinates.

  Args:
    m: Model owning the per-world sleep policies.
    d: Simulation data to update. Disabled DOF velocities and residual forces are cleared.
    world_mask: Optional boolean array of shape ``(nworld,)`` selecting worlds to modify.
    initial_tree_asleep: Optional int32 sleep-cycle snapshot of shape ``(1 or nworld, ntree)``.
      Without a snapshot, enabled trees start awake. Current ALWAYS policies override the snapshot.
  """
  if world_mask is not None and (world_mask.shape != (d.nworld,) or world_mask.dtype != wp.bool):
    raise ValueError("world_mask must be a boolean array of shape (nworld,).")
  if initial_tree_asleep is not None and (
    initial_tree_asleep.shape not in ((1, m.ntree), (d.nworld, m.ntree)) or initial_tree_asleep.dtype != wp.int32
  ):
    raise ValueError("initial_tree_asleep must be an int32 array of shape (1 or nworld, ntree).")
  wp.launch(
    _reset_sleep_trees,
    dim=(d.nworld, m.ntree),
    inputs=[m.tree_sleep_policy, initial_tree_asleep, world_mask],
    outputs=[d.tree_asleep],
  )
  wp.launch(
    _clear_disabled_dofs,
    dim=(d.nworld, m.nv),
    inputs=[m.dof_treeid, m.tree_sleep_policy, world_mask, None],
    outputs=[d.qvel, d.qacc_warmstart, d.qfrc_applied, d.qacc],
  )
  update_sleep(m, d, world_mask=world_mask)


@wp.kernel
def _set_sleep_policy(
  # Model:
  ntree: int,
  tree_sleep_policy: wp.array2d[int],
  # In:
  tree_mask: wp.array2d[int],
  world_mask: wp.array[bool],
  policy: int,
  # Data out:
  tree_asleep_out: wp.array2d[int],
):
  worldid = wp.tid()
  if world_mask and not world_mask[worldid]:
    return
  # Break old cycles before disabling members; otherwise peers retain links into disabled trees.
  for treeid in range(ntree):
    if tree_mask[worldid, treeid] != 0:
      _wake_tree(ntree, tree_sleep_policy, worldid, treeid, K_AWAKE_VAL, tree_asleep_out)
  for treeid in range(ntree):
    if tree_mask[worldid, treeid] != 0:
      # Runtime policy changes intentionally write the model (unsupported by the analyzer).
      tree_sleep_policy[worldid % tree_sleep_policy.shape[0], treeid] = policy  # kernel_analyzer: ignore
      tree_asleep_out[worldid, treeid] = treeid if policy == SleepPolicy.ALWAYS else K_AWAKE_VAL


def set_sleep_policy(
  m: types.Model, d: types.Data, tree_mask: wp.array, policy: SleepPolicy, *, world_mask: wp.array | None = None
):
  """Change selected tree policies and reconcile their sleep cycles immediately.

  Requires sleeping and islands, and a policy table batched to ``nworld``. Disabled trees lose
  residual DOF motion; re-enabled trees start awake. Coordinates are preserved. Deactivate equality
  constraints joining disabled trees separately. This operation supports CUDA graph capture.

  Args:
    m: Model whose ``tree_sleep_policy`` has shape ``(nworld, ntree)``.
    d: Simulation data to update.
    tree_mask: Int32 array of shape ``(nworld, ntree)`` selecting trees.
    policy: Compiled runtime sleep policy to apply.
    world_mask: Optional boolean array of shape ``(nworld,)`` selecting worlds to modify.
  """
  if not (m.opt.enableflags & types.EnableBit.SLEEP) or m.opt.disableflags & types.DisableBit.ISLAND:
    raise ValueError("Changing sleep policy requires sleeping and islands to be enabled.")
  if m.tree_sleep_policy.shape != (d.nworld, m.ntree):
    raise ValueError("Batch tree_sleep_policy to nworld when calling put_model.")
  if tree_mask.shape != (d.nworld, m.ntree) or tree_mask.dtype != wp.int32:
    raise ValueError("tree_mask must be an int32 array of shape (nworld, ntree).")
  if world_mask is not None and (world_mask.shape != (d.nworld,) or world_mask.dtype != wp.bool):
    raise ValueError("world_mask must be a boolean array of shape (nworld,).")
  policy = SleepPolicy(policy)
  wp.launch(
    _set_sleep_policy,
    dim=d.nworld,
    inputs=[m.ntree, m.tree_sleep_policy, tree_mask, world_mask, int(policy)],
    outputs=[d.tree_asleep],
  )
  wp.launch(
    _clear_disabled_dofs,
    dim=(d.nworld, m.nv),
    inputs=[m.dof_treeid, m.tree_sleep_policy, world_mask, tree_mask],
    outputs=[d.qvel, d.qacc_warmstart, d.qfrc_applied, d.qacc],
  )
  update_sleep(m, d, world_mask=world_mask)


@event_scope
def wake_tendon(m: types.Model, d: types.Data):
  """Wakes sleeping trees with constrained tendons."""
  if m.ntendon == 0:
    return

  wp.launch(
    _wake_tendon_kernel,
    dim=(d.nworld, m.ntendon),
    inputs=[
      m.ntree,
      m.ntendon,
      m.body_treeid,
      m.jnt_bodyid,
      m.tree_sleep_policy,
      m.geom_bodyid,
      m.site_bodyid,
      m.tendon_adr,
      m.tendon_num,
      m.tendon_limited,
      m.tendon_range,
      m.tendon_margin,
      m.wrap_type,
      m.wrap_objid,
      d.ten_length,
      d.tree_awake,
    ],
    outputs=[d.tree_asleep],
  )


@event_scope
def wake_equality(m: types.Model, d: types.Data):
  """Wakes sleeping trees connected by equality constraints."""
  if m.neq == 0:
    return

  wp.launch(
    _wake_equality_kernel,
    dim=(d.nworld, m.neq),
    inputs=[
      m.ntree,
      m.neq,
      m.body_treeid,
      m.jnt_bodyid,
      m.tree_sleep_policy,
      m.geom_bodyid,
      m.site_bodyid,
      m.eq_type,
      m.eq_obj1id,
      m.eq_obj2id,
      m.eq_objtype,
      m.tendon_adr,
      m.tendon_num,
      m.wrap_type,
      m.wrap_objid,
      d.eq_active,
      d.tree_awake,
    ],
    outputs=[d.tree_asleep],
  )


@wp.kernel
def _sweep_awake_trees(  # kernel_analyzer: ignore
  # Model:
  nbody: int,
  body_treeid: wp.array[int],
  dof_length: wp.array[float],
  tree_dofadr: wp.array[int],
  tree_dofnum: wp.array[int],
  tree_sleep_policy: wp.array2d[int],
  # Data in:
  qvel_in: wp.array2d[float],
  qfrc_applied_in: wp.array2d[float],
  xfrc_applied_in: wp.array2d[wp.spatial_vector],
  # In:
  opt_sleep_tolerance: wp.array[float],
  # Out:
  tree_asleep_out: wp.array2d[int],  # kernel_analyzer: ignore
):
  worldid, treeid = wp.tid()
  if tree_sleep_policy[worldid % tree_sleep_policy.shape[0], treeid] == SleepPolicy.ALWAYS:
    return

  sleep_tolerance = opt_sleep_tolerance[worldid % opt_sleep_tolerance.shape[0]]
  as_val = tree_asleep_out[worldid, treeid]
  if as_val >= 0:
    return

  if _tree_can_sleep(
    nbody,
    body_treeid,
    dof_length,
    tree_dofadr,
    tree_dofnum,
    tree_sleep_policy,
    qvel_in,
    qfrc_applied_in,
    xfrc_applied_in,
    worldid,
    treeid,
    sleep_tolerance,
  ):
    if as_val < -1:
      tree_asleep_out[worldid, treeid] = as_val + 1
  else:
    tree_asleep_out[worldid, treeid] = K_AWAKE_VAL


@wp.kernel
def _check_island_can_sleep(
  # Model:
  ntree: int,
  # Data in:
  nisland_in: wp.array[int],
  tree_asleep_in: wp.array2d[int],
  tree_island_in: wp.array2d[int],
  # Out:
  island_can_sleep_out: wp.array2d[int],
):
  worldid, treeid = wp.tid()
  nisland = nisland_in[worldid]
  island_id = tree_island_in[worldid, treeid]
  if island_id >= 0 and island_id < nisland:
    as_val = tree_asleep_in[worldid, treeid]
    if as_val < -1:
      # Not ready to sleep yet
      wp.atomic_min(island_can_sleep_out, worldid, island_id, 0)


@wp.kernel
def _build_cycles(  # kernel_analyzer: ignore
  # Model:
  ntree: int,
  tree_dofadr: wp.array[int],
  tree_dofnum: wp.array[int],
  # Data in:
  nisland_in: wp.array[int],
  tree_island_in: wp.array2d[int],
  # In:
  island_can_sleep_in: wp.array2d[int],
  # Out:
  tree_asleep_out: wp.array2d[int],  # kernel_analyzer: ignore
  qvel_out: wp.array2d[float],  # kernel_analyzer: ignore
  qacc_out: wp.array2d[float],  # kernel_analyzer: ignore
):
  worldid = wp.tid()

  num_islands = nisland_in[worldid]
  for island_id in range(num_islands):
    if island_can_sleep_in[worldid, island_id] == 1:
      first_tree = int(-1)
      prev_tree = int(-1)
      for t in range(ntree):
        if tree_island_in[worldid, t] == island_id:
          if first_tree == -1:
            first_tree = t
          if prev_tree != -1:
            tree_asleep_out[worldid, prev_tree] = t
          prev_tree = t

          # Zero velocities and accelerations
          dofadr = tree_dofadr[t]
          dofnum = tree_dofnum[t]
          for d in range(dofnum):
            qvel_out[worldid, dofadr + d] = 0.0
            qacc_out[worldid, dofadr + d] = 0.0

      if first_tree != -1:
        tree_asleep_out[worldid, prev_tree] = first_tree

  # Sleep unconstrained trees
  for t in range(ntree):
    island_id = tree_island_in[worldid, t]
    if island_id < 0 or island_id >= num_islands:
      if tree_asleep_out[worldid, t] == -1:
        tree_asleep_out[worldid, t] = t  # self-cycle

      # Ensure sleeping tree dof velocity and acceleration remain exactly zero
      if tree_asleep_out[worldid, t] >= 0:
        dofadr = tree_dofadr[t]
        dofnum = tree_dofnum[t]
        for d in range(dofnum):
          # TODO(team): shouldn't be necessary to zero if island is already asleep
          qvel_out[worldid, dofadr + d] = 0.0
          qacc_out[worldid, dofadr + d] = 0.0


@event_scope
def sleep(m: types.Model, d: types.Data):
  """Puts trees to sleep according to velocity tolerance."""
  # 1. Sweep over awake trees and increment counter if they can sleep
  wp.launch(
    _sweep_awake_trees,
    dim=(d.nworld, m.ntree),
    inputs=[
      m.nbody,
      m.body_treeid,
      m.dof_length,
      m.tree_dofadr,
      m.tree_dofnum,
      m.tree_sleep_policy,
      d.qvel,
      d.qfrc_applied,
      d.xfrc_applied,
      m.opt.sleep_tolerance,
    ],
    outputs=[d.tree_asleep],
  )

  # 2. Check which constraint islands can sleep (all trees in island must be asleep)
  island_can_sleep = wp.ones((d.nworld, m.ntree), dtype=int)
  wp.launch(
    _check_island_can_sleep,
    dim=(d.nworld, m.ntree),
    inputs=[
      m.ntree,
      d.nisland,
      d.tree_asleep,
      d.tree_island,
    ],
    outputs=[island_can_sleep],
  )

  # 3. Build sleep cycles for sleeping islands and sleep unconstrained trees
  wp.launch(
    _build_cycles,
    dim=d.nworld,
    inputs=[
      m.ntree,
      m.tree_dofadr,
      m.tree_dofnum,
      d.nisland,
      d.tree_island,
      island_can_sleep,
    ],
    outputs=[
      d.tree_asleep,
      d.qvel,
      d.qacc,
    ],
  )
