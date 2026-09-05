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

"""Tests for the fused per-world forward kernels (fused_world.py)."""

import dataclasses

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest

import mujoco_warp as mjw
from mujoco_warp import test_data
from mujoco_warp._src import collision_driver
from mujoco_warp._src import fused_world
from mujoco_warp._src import sleep
from mujoco_warp._src import smooth
from mujoco_warp._src.types import ConstraintType
from mujoco_warp._src.types import DisableBit
from mujoco_warp._src.types import OverflowType
from mujoco_warp._src.types import SleepState

# fused and stock paths differ only in floating point summation / factorization order
_RTOL = 1e-5
_ATOL = 1e-6


def _option(contact: bool) -> str:
  contact_flag = "enable" if contact else "disable"
  return f"""
  <option integrator="implicitfast" cone="pyramidal" solver="Newton" jacobian="sparse" sleep_tolerance="0.01"
          gravity="0 0 -9.81">
    <flag sleep="enable" island="enable" contact="{contact_flag}"/>
  </option>
"""


def factory_like_xml(
  n_free: int = 19, n_fixed: int = 19, n_hover: int = 3, contact: bool = False, equality: bool = False
) -> str:
  """Franka-like 9-DOF chain (11 bodies) with servos, limits and gravcomp, plus free/static bodies.

  ``n_hover`` of the free bodies carry gravcomp=1 so that they hover with zero velocity and fall
  asleep through the MJ_MINAWAKE countdown during a trajectory. With ``contact`` the free bodies
  rest on a floor plane with condim 1/3/4/6 cycling over the bodies, one arm joint has friction loss
  and the static bodies are collision-free sockets; ``equality`` adds a two-joint and a one-joint
  JOINT equality on the arm.
  """
  j3_extra = ' frictionloss="0.4"' if contact else ""
  chain = f"""
    <body name="link0" pos="0 0 0">
      <geom type="box" size="0.05 0.05 0.05" mass="3"/>
      <body name="link1" pos="0 0 0.15" gravcomp="1"><joint name="j1" type="hinge" axis="0 0 1" range="-2.9 2.9" damping="0.5" armature="0.1"/><geom type="capsule" size="0.04" fromto="0 0 0 0 0 0.2" mass="3"/>
      <body name="link2" pos="0 0 0.2" gravcomp="1"><joint name="j2" type="hinge" axis="0 1 0" range="-1.7 1.7" damping="0.5" armature="0.1" stiffness="2"/><geom type="capsule" size="0.04" fromto="0 0 0 0 0 0.2" mass="3"/>
      <body name="link3" pos="0 0 0.2" gravcomp="1"><joint name="j3" type="hinge" axis="0 0 1" range="-2.9 2.9" damping="0.3" armature="0.05"{j3_extra}/><geom type="capsule" size="0.04" fromto="0 0 0 0 0 0.2" mass="2.5"/>
      <body name="link4" pos="0 0 0.2" gravcomp="1"><joint name="j4" type="hinge" axis="0 -1 0" range="-3.0 -0.1" damping="0.3" armature="0.05"/><geom type="capsule" size="0.04" fromto="0 0 0 0 0 0.2" mass="2.5"/>
      <body name="link5" pos="0 0 0.2" gravcomp="1"><joint name="j5" type="hinge" axis="0 0 1" range="-2.9 2.9" damping="0.2" armature="0.02"/><geom type="capsule" size="0.03" fromto="0 0 0 0 0 0.2" mass="2"/>
      <body name="link6" pos="0 0 0.2" gravcomp="1"><joint name="j6" type="hinge" axis="0 -1 0" range="-0.1 3.7" damping="0.2" armature="0.02"/><geom type="capsule" size="0.03" fromto="0 0 0 0 0 0.1" mass="1.5"/>
      <body name="link7" pos="0 0 0.1" gravcomp="1"><joint name="j7" type="hinge" axis="0 0 1" range="-2.9 2.9" damping="0.1" armature="0.01"/><geom type="cylinder" size="0.03 0.03" mass="0.5"/>
      <body name="hand" pos="0 0 0.06" gravcomp="1"><geom type="box" size="0.04 0.02 0.03" mass="0.7"/>
        <body name="finger_l" pos="0.03 0 0.05" gravcomp="1"><joint name="fl" type="slide" axis="1 0 0" range="0 0.04" damping="1" armature="0.01"/><geom type="box" size="0.01 0.01 0.02" mass="0.1"/></body>
        <body name="finger_r" pos="-0.03 0 0.05" gravcomp="1"><joint name="fr" type="slide" axis="-1 0 0" range="0 0.04" damping="1" armature="0.01"/><geom type="box" size="0.01 0.01 0.02" mass="0.1"/></body>
      </body></body></body></body></body></body></body></body>
    </body>
  """
  # alternate MuJoCo "simple" bodies (centered cubes: diagonal inertia blocks) and bodies with an
  # offset, rotated inertial frame (dense 6x6 inertia blocks, like Factory's held parts)
  condims = (1, 3, 4, 6)
  free = ""
  for i in range(n_free):
    condim = f' condim="{condims[i % 4]}" friction="{0.6 + 0.1 * (i % 3)} 0.01 0.002"' if contact else ""
    simple_geom = f'<geom type="box" size="0.02 0.02 0.02" mass="0.05"{condim}/>'
    offset_geom = f'<geom type="box" size="0.02 0.03 0.04" pos="0.01 0.005 0.0" euler="0.3 0.2 0.1" mass="0.05"{condim}/>'
    z = 0.019 if contact and i >= n_hover else 0.3
    free += (
      f'<body name="free{i}" pos="{0.5 + 0.1 * i} 0 {z}" gravcomp="{1 if i < n_hover else 0}">'
      f"<freejoint/>{simple_geom if i % 2 == 0 else offset_geom}</body>\n"
    )
  socket_extra = ' contype="0" conaffinity="0"' if contact else ""
  fixed = "".join(
    f'<body name="fixed{i}" pos="{0.5 + 0.1 * i} 0.3 0"><geom type="box" size="0.03 0.03 0.01"{socket_extra}/></body>\n'
    for i in range(n_fixed)
  )
  # the floor's condim 1 lets each body's condim decide (MuJoCo takes the max at equal priority)
  floor = '<geom name="floor" type="plane" size="5 5 0.1" condim="1"/>' if contact else ""
  actuators = "".join(
    f'<position joint="j{i}" kp="{50.0 + 10.0 * i}" kv="{2.0 + i}" ctrlrange="-2.5 2.5" forcerange="-80 80"/>\n'
    for i in range(1, 8)
  )
  actuators += '<position joint="fl" kp="200" ctrlrange="0 0.04"/>\n<position joint="fr" kp="200" ctrlrange="0 0.04"/>\n'
  actuators += "".join(f'<velocity joint="j{i}" kv="{0.5 * i}" ctrlrange="-1 1"/>\n' for i in range(1, 8))
  equalities = ""
  if equality:
    equalities = """
    <equality>
      <joint joint1="j5" joint2="j7" polycoef="0.1 0.5 0.2 0 0" solref="0.02 1"/>
      <joint joint1="fr" polycoef="0.01 0 0 0 0"/>
    </equality>
    """
  return f"""
  <mujoco>
    {_option(contact)}
    <worldbody>
      {floor}
      {chain}
      {free}
      {fixed}
    </worldbody>
    <actuator>
      {actuators}
    </actuator>
    {equalities}
  </mujoco>
  """


def _random_state(
  mjm: mujoco.MjModel, nworld: int, rng: np.random.Generator, qvel_scale: float = 1.0, free_pos_noise: float = 0.2
):
  """Per-world random qpos (unit quaternions for free joints), qvel and ctrl."""
  qpos = np.tile(mjm.qpos0, (nworld, 1))
  qvel = rng.uniform(-qvel_scale, qvel_scale, size=(nworld, mjm.nv))
  for jntid in range(mjm.njnt):
    adr = mjm.jnt_qposadr[jntid]
    if mjm.jnt_type[jntid] == mujoco.mjtJoint.mjJNT_FREE:
      qpos[:, adr : adr + 3] += rng.uniform(-free_pos_noise, free_pos_noise, size=(nworld, 3))
      quat = rng.normal(size=(nworld, 4))
      qpos[:, adr + 3 : adr + 7] = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    else:
      lo, hi = mjm.jnt_range[jntid]
      # some worlds sit outside the limits so that limit rows are active
      qpos[:, adr] = rng.uniform(lo - 0.2, hi + 0.2, size=nworld)
  ctrl = rng.uniform(-1.0, 1.0, size=(nworld, mjm.nu))
  return qpos.astype(np.float32), qvel.astype(np.float32), ctrl.astype(np.float32)


def _resting_state(mjm: mujoco.MjModel, nworld: int, rng: np.random.Generator):
  """Free bodies resting on the floor (tiny pose noise) so that contacts of every condim exist."""
  qpos = np.tile(mjm.qpos0, (nworld, 1))
  qvel = rng.uniform(-0.05, 0.05, size=(nworld, mjm.nv))
  for jntid in range(mjm.njnt):
    adr = mjm.jnt_qposadr[jntid]
    if mjm.jnt_type[jntid] == mujoco.mjtJoint.mjJNT_FREE:
      qpos[:, adr : adr + 2] += rng.uniform(-0.01, 0.01, size=(nworld, 2))
      qpos[:, adr + 2] += rng.uniform(-0.004, 0.0, size=nworld)
      axis = rng.normal(size=(nworld, 3))
      axis /= np.linalg.norm(axis, axis=1, keepdims=True)
      angle = rng.uniform(-0.05, 0.05, size=(nworld, 1))
      qpos[:, adr + 3 : adr + 4] = np.cos(angle / 2)
      qpos[:, adr + 4 : adr + 7] = axis * np.sin(angle / 2)
    else:
      lo, hi = mjm.jnt_range[jntid]
      qpos[:, adr] = rng.uniform(lo - 0.1, hi + 0.1, size=nworld)
  ctrl = rng.uniform(-1.0, 1.0, size=(nworld, mjm.nu))
  return qpos.astype(np.float32), qvel.astype(np.float32), ctrl.astype(np.float32)


def _share_contacts(m, d_src, d_dst):
  """Collide d_src with MJWarp (as an external provider would) and copy the contacts to d_dst."""
  m.opt.run_collision_detection = True
  try:
    collision_driver.collision(m, d_src)
  finally:
    m.opt.run_collision_detection = False
  wp.synchronize()
  for f in dataclasses.fields(d_src.contact):
    src = getattr(d_src.contact, f.name)
    if isinstance(src, wp.array):
      wp.copy(getattr(d_dst.contact, f.name), src)
  wp.copy(d_dst.nacon, d_src.nacon)
  wp.copy(d_dst.ncollision, d_src.ncollision)
  wp.synchronize()


def _rows(m, d, w: int):
  """Constraint rows of world w keyed by (type, id, dim) with their scalar and Jacobian values."""
  nefc = min(int(d.nefc.numpy()[w]), d.njmax)
  efc = d.efc
  typ = efc.type.numpy()[w]
  ids = efc.id.numpy()[w]
  rownnz = efc.J_rownnz.numpy()[w]
  rowadr = efc.J_rowadr.numpy()[w]
  colind = efc.J_colind.numpy()[w, 0]
  J = efc.J.numpy()[w, 0]
  efc_address = d.contact.efc_address.numpy()
  scalars = {name: getattr(efc, name).numpy()[w] for name in ("pos", "margin", "D", "aref", "vel", "frictionloss")}
  rows = {}
  for efcid in range(nefc):
    t, i = int(typ[efcid]), int(ids[efcid])
    dim = 0
    if t in (ConstraintType.CONTACT_FRICTIONLESS, ConstraintType.CONTACT_PYRAMIDAL):
      dim = efcid - int(efc_address[i, 0])
    key = (t, i, dim)
    assert key not in rows, f"duplicate row {key}"
    jac = {int(colind[rowadr[efcid] + k]): float(J[rowadr[efcid] + k]) for k in range(rownnz[efcid])}
    rows[key] = ({name: float(values[efcid]) for name, values in scalars.items()}, jac, efcid)
  return rows


def _jtdaj_blocks(d, rows, w: int):
  """Set of (first row key, nrow) of the jtdaj blocks of world w."""
  by_efcid = {efcid: key for key, (_, _, efcid) in rows.items()}
  nblock = int(d.efc.jtdaj_nblock.numpy()[w])
  adr = d.efc.jtdaj_adr.numpy()[w]
  nrow = d.efc.jtdaj_nrow.numpy()[w]
  return {(by_efcid[int(adr[b])], int(nrow[b])) for b in range(nblock)}


class FusedWorldTest(absltest.TestCase):
  NWORLD = 8

  def _make(
    self,
    seed: int,
    qvel_scale: float = 1.0,
    contact: bool = False,
    equality: bool = False,
    resting: bool = False,
    **data_kwargs,
  ):
    mjm, mjd, m, _ = test_data.fixture(xml=factory_like_xml(contact=contact, equality=equality), nworld=1)
    m.opt.run_collision_detection = False
    if not wp.get_device().is_cuda:
      self.skipTest("fused world kernels only run on CUDA")
    self.assertEqual(m.nv, 9 + 6 * 19)
    self.assertEqual(m.nbody, 1 + 11 + 19 + 19)
    self.assertEqual(m.ntree, 20)
    self.assertEqual(m.nu, 16)
    # both factor paths are exercised: compact (diagonal) blocks and dense scalar Cholesky blocks
    block_adr = m.qLD_block_adr.numpy()[m.tree_dofadr.numpy()]
    self.assertGreater((block_adr == -2).sum(), 0)
    self.assertGreater((block_adr >= 0).sum(), 1)

    rng = np.random.default_rng(seed)
    if resting:
      qpos, qvel, ctrl = _resting_state(mjm, self.NWORLD, rng)
    else:
      qpos, qvel, ctrl = _random_state(mjm, self.NWORLD, rng, qvel_scale)
    qfrc_applied = np.zeros((self.NWORLD, mjm.nv), dtype=np.float32)
    xfrc_applied = np.zeros((self.NWORLD, mjm.nbody, 6), dtype=np.float32)
    # applied forces in a few worlds: generalized on the arm, Cartesian on a free body and the hand
    qfrc_applied[1, :7] = rng.uniform(-1.0, 1.0, size=7)
    xfrc_applied[1, 16] = rng.uniform(-1.0, 1.0, size=6)
    xfrc_applied[6, 9] = rng.uniform(-2.0, 2.0, size=6)

    if data_kwargs:
      # small capacities: start from an empty MjData so that put_data's capacity checks pass
      mjd = mujoco.MjData(mjm)
    datas = []
    for _ in range(2):
      d = mjw.put_data(mjm, mjd, nworld=self.NWORLD, **data_kwargs)
      d.qpos.assign(qpos)
      d.qvel.assign(qvel)
      d.ctrl.assign(ctrl)
      d.qfrc_applied.assign(qfrc_applied)
      d.xfrc_applied.assign(xfrc_applied)
      datas.append(d)
    return mjm, m, datas

  def _induce_sleep(self, m, datas):
    """Put a few free-body trees to sleep (cycles) and create wake conditions in other worlds."""
    d0 = datas[0]
    tree_asleep = d0.tree_asleep.numpy()
    qvel = d0.qvel.numpy()
    tree_dofadr = m.tree_dofadr.numpy()
    tree_dofnum = m.tree_dofnum.numpy()

    def quiet(w, t):
      qvel[w, tree_dofadr[t] : tree_dofadr[t] + tree_dofnum[t]] = 0.0

    # world 2: two self-cycles; world 3: a two-tree cycle; both stay asleep
    for t in (1, 2):
      tree_asleep[2, t] = t
      quiet(2, t)
    tree_asleep[3, 3] = 4
    tree_asleep[3, 4] = 3
    quiet(3, 3)
    quiet(3, 4)
    # world 4: asleep tree with nonzero velocity -> woken by sleep.wake
    tree_asleep[4, 5] = 5
    # world 5: asleep tree whose tree_awake flag mismatches -> woken by sleep.wake
    tree_asleep[5, 6] = 6
    quiet(5, 6)
    for d in datas:
      d.tree_asleep.assign(tree_asleep)
      d.qvel.assign(qvel)
      sleep.update_sleep(m, d)
      tree_awake = d.tree_awake.numpy()
      tree_awake[5, 6] = 1
      d.tree_awake.assign(tree_awake)

  def _run_forward(self, m, d, fused: bool):
    m.opt.fused_world = fused
    try:
      self.assertEqual(fused_world.fused_world(m, d), fused)
      mjw.forward(m, d)
      wp.synchronize()
    finally:
      m.opt.fused_world = True

  def _run_step(self, m, d, fused: bool, finalize: bool = True):
    m.opt.fused_world = fused
    try:
      self.assertEqual(fused_world.fused_world(m, d), fused)
      if finalize:
        mjw.step(m, d)
      else:
        mjw._step_intermediate(m, d)
      wp.synchronize()
    finally:
      m.opt.fused_world = True

  def _assert_close(self, name, fused, ref, rtol=_RTOL, atol=_ATOL):
    fused = np.asarray(fused)
    ref = np.asarray(ref)
    self.assertTrue(np.isfinite(fused).all(), f"{name} not finite")
    np.testing.assert_allclose(fused, ref, rtol=rtol, atol=atol, err_msg=f"mismatch: {name}")

  def _assert_close_scaled(self, name, fused, ref, rel: float):
    """Max abs difference relative to the largest reference magnitude (solver-limited fields)."""
    fused = np.asarray(fused)
    ref = np.asarray(ref)
    self.assertTrue(np.isfinite(fused).all(), f"{name} not finite")
    scale = max(float(np.abs(ref).max()), 1e-6)
    diff = float(np.abs(fused - ref).max()) / scale
    self.assertLessEqual(diff, rel, f"{name}: max diff {diff:.3e} of scale {scale:.3e} exceeds {rel:.1e}")
    return diff

  def _assert_int_equal(self, d_fused, d_ref, names):
    for name in names:
      np.testing.assert_array_equal(getattr(d_fused, name).numpy(), getattr(d_ref, name).numpy(), err_msg=f"mismatch: {name}")

  def _assert_awake_sets_equal(self, d_fused, d_ref):
    """The awake index lists are order-free (stock: atomic order, fused: body/dof order)."""
    for name, count in (("body_awake_ind", "nbody_awake"), ("dof_awake_ind", "nv_awake")):
      n = getattr(d_ref, count).numpy()
      for w in range(d_ref.nworld):
        ref_set = set(getattr(d_ref, name).numpy()[w, : n[w]].tolist())
        fused_ind = getattr(d_fused, name).numpy()[w, : n[w]]
        self.assertEqual(set(fused_ind.tolist()), ref_set, f"mismatch: {name} world {w}")

  def _assert_rows_equal(self, m, d_fused, d_ref, rtol=_RTOL, atol=_ATOL):
    """Per world, the row multisets (values, Jacobians, jtdaj blocks, contact addresses) agree."""
    self._assert_int_equal(d_fused, d_ref, ("ne", "nf", "nl", "nefc"))
    np.testing.assert_array_equal(d_fused.efc.jtdaj_nblock.numpy(), d_ref.efc.jtdaj_nblock.numpy(), err_msg="jtdaj_nblock")
    efc_address_fused = d_fused.contact.efc_address.numpy()
    efc_address_ref = d_ref.contact.efc_address.numpy()
    np.testing.assert_array_equal(efc_address_fused < 0, efc_address_ref < 0, err_msg="contact.efc_address validity")
    max_diff = {}
    for w in range(d_ref.nworld):
      rows_ref = _rows(m, d_ref, w)
      rows_fused = _rows(m, d_fused, w)
      self.assertEqual(sorted(rows_fused), sorted(rows_ref), f"row keys differ in world {w}")
      for key, (scal_ref, jac_ref, _) in rows_ref.items():
        scal_fused, jac_fused, _ = rows_fused[key]
        for name in scal_ref:
          np.testing.assert_allclose(
            scal_fused[name], scal_ref[name], rtol=rtol, atol=atol, err_msg=f"row {key} {name} world {w}"
          )
          max_diff[name] = max(max_diff.get(name, 0.0), abs(scal_fused[name] - scal_ref[name]))
        self.assertEqual(sorted(jac_fused), sorted(jac_ref), f"row {key} colind world {w}")
        for col, value in jac_ref.items():
          np.testing.assert_allclose(jac_fused[col], value, rtol=rtol, atol=atol, err_msg=f"row {key} J[{col}] world {w}")
          max_diff["J"] = max(max_diff.get("J", 0.0), abs(jac_fused[col] - value))
      self.assertEqual(_jtdaj_blocks(d_fused, rows_fused, w), _jtdaj_blocks(d_ref, rows_ref, w), f"jtdaj blocks world {w}")
    return max_diff

  def _dense_m(self, m, d):
    """Per-world dense float64 inertia matrices from the sparse CSR M."""
    rowadr = m.M_rowadr.numpy()
    rownnz = m.M_rownnz.numpy()
    colind = m.M_colind.numpy()
    M = d.M.numpy().astype(np.float64)
    dense = np.zeros((d.nworld, m.nv, m.nv))
    for i in range(m.nv):
      for k in range(rownnz[i]):
        j = colind[rowadr[i] + k]
        dense[:, i, j] = M[:, rowadr[i] + k]
        dense[:, j, i] = M[:, rowadr[i] + k]
    return dense

  def _assert_solve_accuracy(self, name, m, d_ref, d_fused, rhs, x_ref, x_fused):
    """Both fp32 solves must be within the fp32 conditioning envelope of a float64 reference."""
    dense = self._dense_m(m, d_ref)
    x64 = np.stack([np.linalg.solve(dense[w], rhs[w].astype(np.float64)) for w in range(d_ref.nworld)])
    scale = np.abs(x64).max()
    err_ref = np.abs(x_ref - x64).max() / scale
    err_fused = np.abs(x_fused - x64).max() / scale
    print(f"{name}: max rel error vs float64: stock {err_ref:.3e} fused {err_fused:.3e}")
    # the scalar Cholesky of the 9-dof block measures 7.4e-7 (tile Cholesky: 8e-8); bound at 2e-6
    self.assertLessEqual(err_fused, 2e-6, f"{name}: fused solve less accurate than expected")
    self.assertLessEqual(err_ref, 2e-6, f"{name}: stock solve less accurate than expected")
    self._assert_close(name, x_fused, x_ref, rtol=1e-3, atol=1e-4 * scale)

  def _qld_upper_mask(self, m):
    """Mask of the packed upper-triangular factor entries (the lower triangle is unspecified)."""
    mask = np.zeros(m.qLD_block_total, dtype=bool)
    block_adr = m.qLD_block_adr.numpy()
    for start, num in zip(m.tree_dofadr.numpy(), m.tree_dofnum.numpy()):
      adr = block_adr[start]
      if adr < 0:
        continue
      for i in range(num):
        for j in range(i, num):
          mask[adr + i * num + j] = True
    return mask

  def test_predicate(self):
    """The predicate accepts the Factory-like model and rejects unsupported features."""
    _, m, datas = self._make(seed=0)
    d = datas[0]
    self.assertTrue(fused_world.fused_world(m, d))
    # runtime toggles are re-evaluated on every call
    m.sensor_rne_postconstraint = True
    self.assertFalse(fused_world.fused_world(m, d))
    m.sensor_rne_postconstraint = False
    self.assertTrue(fused_world.fused_world(m, d))
    m.opt.run_collision_detection = True
    self.assertFalse(fused_world.fused_world(m, d))
    m.opt.run_collision_detection = False
    # host switch
    m.opt.fused_world = False
    self.assertFalse(fused_world.fused_world(m, d))
    m.opt.fused_world = True
    self.assertTrue(fused_world.fused_world(m, d))

    # structural rejections: ball joint, tendon
    for extra in (
      '<body pos="3 0 1"><joint type="ball"/><geom type="sphere" size="0.05"/></body>',
      '<body pos="3 0 1"><joint name="s0" type="slide"/><geom type="sphere" size="0.05"/></body>'
      '</worldbody><tendon><fixed><joint joint="s0" coef="1"/></fixed></tendon><worldbody>',
    ):
      xml = factory_like_xml().replace("</worldbody>", extra + "</worldbody>", 1)
      mjm2, mjd2, m2, d2 = test_data.fixture(xml=xml, nworld=1)
      m2.opt.run_collision_detection = False
      self.assertFalse(fused_world.fused_world(m2, d2))

  def test_variants_match_stock(self):
    """Model variants the predicate admits behave like stock.

    Mocap body, actuator gravcomp and force range, a second wide tree, no actuators, disabled
    spring/damper/gravity/actuation, an nvmax overflow and a 12-level chain.
    """
    mocap = '<body name="mocap0" mocap="true" pos="1 1 0.5"><geom type="sphere" size="0.02"/></body>'
    second_chain = (
      "".join(
        f'<body pos="0 0 0.1"><joint type="hinge" axis="{"0 1 0" if i % 2 else "1 0 0"}" range="-1 1" damping="0.2"/>'
        f'<geom type="capsule" size="0.02" fromto="0 0 0 0 0 0.1" mass="0.5"/>'
        for i in range(7)
      )
      + "</body>" * 7
    )
    second_tree = f'<body name="chain2" pos="-1 -1 0"><geom type="box" size="0.03 0.03 0.03" mass="1"/>{second_chain}</body>'
    variants = {
      "mocap": (factory_like_xml().replace("</worldbody>", mocap + "</worldbody>", 1), 0, None),
      "actgravcomp": (
        factory_like_xml()
        .replace(
          '<joint name="j1" type="hinge"', '<joint name="j1" type="hinge" actuatorgravcomp="true" actuatorfrcrange="-30 30"', 1
        )
        .replace("<actuator>", '<option><flag actuation="enable"/></option><actuator>', 1)
        .replace('<option><flag actuation="enable"/></option>', "", 1),
        0,
        None,
      ),
      "second_wide_tree": (factory_like_xml(n_free=10).replace("</worldbody>", second_tree + "</worldbody>", 1), 0, None),
      "no_actuators": (factory_like_xml().split("<actuator>")[0] + "</mujoco>", 0, None),
      "disable_flags": (
        factory_like_xml(),
        DisableBit.SPRING | DisableBit.DAMPER | DisableBit.GRAVITY | DisableBit.ACTUATION,
        None,
      ),
      # only trees with constraint rows count towards nvmax (the arm, when one of its limits is hit)
      "nvmax_overflow": (factory_like_xml(), 0, 4),
      # a fixed link8 between link7 and the hand: 12 tree levels like Factory's Franka, one deeper
      # than the fixture, so the ancestor walks run past the depth the other variants exercise
      "deep_chain": (
        factory_like_xml()
        .replace(
          '<body name="hand" pos="0 0 0.06" gravcomp="1">',
          '<body name="link8" pos="0 0 0.03"><body name="hand" pos="0 0 0.03" gravcomp="1">',
          1,
        )
        .replace("</body>" * 8 + "\n    </body>", "</body>" * 9 + "\n    </body>", 1),
        0,
        None,
      ),
    }
    exact_int = (
      "tree_asleep",
      "tree_awake",
      "body_awake",
      "ntree_awake",
      "nbody_awake",
      "nv_awake",
      "ncdof",
      "nsingleton6",
      "overflow",
      "nefc",
      "nisland",
      "tree_island",
    )
    for name, (xml, disableflags, nvmax) in variants.items():
      with self.subTest(variant=name):
        mjm, mjd, m, _ = test_data.fixture(xml=xml, nworld=1)
        m.opt.run_collision_detection = False
        m.opt.disableflags = m.opt.disableflags | disableflags
        rng = np.random.default_rng(11)
        qpos, qvel, ctrl = _random_state(mjm, 4, rng, qvel_scale=0.5)
        datas = []
        for _ in range(2):
          kwargs = {"nvmax": nvmax} if nvmax is not None else {}
          d = mjw.put_data(mjm, mjd, nworld=4, **kwargs)
          d.qpos.assign(qpos)
          d.qvel.assign(qvel)
          d.ctrl.assign(ctrl)
          if m.nmocap:
            d.mocap_pos.assign(np.tile(np.array([[1.1, 0.9, 0.6]], dtype=np.float32), (4, m.nmocap, 1)))
          datas.append(d)
        d_ref, d_fused = datas
        self.assertTrue(fused_world.fused_world(m, d_fused), name)
        if name == "second_wide_tree":
          self.assertEqual((m.tree_dofnum.numpy() > fused_world.NVTREE_SMALL).sum(), 2)
        if name == "no_actuators":
          self.assertEqual(m.nu, 0)
        if name == "deep_chain":
          self.assertEqual(len(m.body_tree), 12)
        for _ in range(3):
          self._run_step(m, d_ref, fused=False)
          self._run_step(m, d_fused, fused=True)
        self._assert_int_equal(d_fused, d_ref, exact_int)
        if name == "nvmax_overflow":
          self.assertTrue((d_ref.overflow.numpy() & OverflowType.NVMAX).any())
        else:
          self.assertEqual(int(d_ref.overflow.numpy().max()), 0)
          for field in ("qpos", "qvel", "qacc_warmstart", "xpos", "cvel", "qfrc_bias", "qfrc_passive", "qfrc_actuator"):
            self._assert_close_scaled(field, getattr(d_fused, field).numpy(), getattr(d_ref, field).numpy(), rel=1e-4)
        if name == "mocap":
          self.assertGreater(m.nmocap, 0)
          self.assertTrue((d_ref.body_awake.numpy()[:, -1] == SleepState.AWAKE).all())

  def test_forward_matches_stock(self):
    """One forward pass: fused kernels reproduce every replaced Data field of the stock launches."""
    mjm, m, datas = self._make(seed=1)
    self._induce_sleep(m, datas)
    d_ref, d_fused = datas
    self._run_forward(m, d_ref, fused=False)
    self._run_forward(m, d_fused, fused=True)

    exact_int = (
      "tree_asleep",
      "tree_awake",
      "body_awake",
      "ntree_awake",
      "nbody_awake",
      "nv_awake",
      "ncdof",
      "nsingleton6",
      "dof_cdof",
      "cdof_dof",
      "nisland",
      "tree_island",
      "island_nv",
      "ne",
      "nf",
      "nl",
      "nefc",
      "overflow",
    )
    self._assert_int_equal(d_fused, d_ref, exact_int)

    # a few worlds must actually be asleep / woken for the comparison to be meaningful
    tree_awake = d_fused.tree_awake.numpy()
    self.assertEqual(tree_awake[2, 1], 0)
    self.assertEqual(tree_awake[2, 2], 0)
    self.assertEqual(tree_awake[3, 3], 0)
    self.assertEqual(tree_awake[3, 4], 0)
    self.assertEqual(tree_awake[4, 5], 1)
    self.assertEqual(tree_awake[5, 6], 1)
    self.assertLess(d_fused.tree_asleep.numpy()[4, 5], 0)
    self._assert_awake_sets_equal(d_fused, d_ref)

    float_fields = (
      "xpos",
      "xquat",
      "xmat",
      "xipos",
      "ximat",
      "xanchor",
      "xaxis",
      "geom_xpos",
      "geom_xmat",
      "subtree_com",
      "cinert",
      "cdof",
      "crb",
      "M",
      "actuator_length",
      "actuator_velocity",
      "actuator_force",
      "qfrc_actuator",
      "cvel",
      "cdof_dot",
      "qfrc_spring",
      "qfrc_damper",
      "qfrc_gravcomp",
      "qfrc_passive",
      "cacc",
      "cfrc_int",
      "qfrc_bias",
      "qfrc_smooth",
    )
    for name in float_fields:
      # crb's translational components of a subtree root cancel analytically; the ~1e-6 residual
      # depends on the summation order (stock atomics vs fused gather)
      atol = 1e-5 if name == "crb" else _ATOL
      self._assert_close(name, getattr(d_fused, name).numpy(), getattr(d_ref, name).numpy(), atol=atol)

    # gravcomp and limits are exercised; limit rows agree row by row
    self.assertGreater(np.abs(d_ref.qfrc_gravcomp.numpy()).max(), 1.0)
    self.assertGreater(d_ref.nl.numpy().max(), 0)
    self.assertGreater(np.abs(d_ref.qfrc_applied.numpy()).max(), 0.0)
    self._assert_rows_equal(m, d_fused, d_ref)

    # actuator moments: the stock row order is atomic-dependent, compare per actuator
    rownnz = d_ref.moment_rownnz.numpy()
    np.testing.assert_array_equal(d_fused.moment_rownnz.numpy(), rownnz)
    for w in range(self.NWORLD):
      for u in range(m.nu):
        self.assertEqual(rownnz[w, u], 1)
        ref_adr = d_ref.moment_rowadr.numpy()[w, u]
        fused_adr = d_fused.moment_rowadr.numpy()[w, u]
        self.assertEqual(d_fused.moment_colind.numpy()[w, fused_adr], d_ref.moment_colind.numpy()[w, ref_adr])
        self.assertEqual(d_fused.actuator_moment.numpy()[w, fused_adr], d_ref.actuator_moment.numpy()[w, ref_adr])

    # inertia factor: compare the packed upper triangles and the solve they produce
    mask = self._qld_upper_mask(m)
    self.assertTrue(mask.any())
    qld_fused = d_fused.qLD.numpy()[:, mask]
    qld_ref = d_ref.qLD.numpy()[:, mask]
    self._assert_close("qLD", qld_fused, qld_ref, rtol=1e-4, atol=1e-5)
    rhs = np.random.default_rng(3).normal(size=(self.NWORLD, m.nv)).astype(np.float32)
    sols = []
    for d in datas:
      x = wp.zeros((self.NWORLD, m.nv), dtype=float)
      smooth.solve_m(m, d, x, wp.array(rhs))
      sols.append(x.numpy())
    self._assert_solve_accuracy("solve_m", m, d_ref, d_fused, rhs, sols[0], sols[1])
    # qacc_smooth: the 9-dof block uses tile Cholesky in stock and scalar Cholesky in the fused
    # path; dofs of sleeping trees are frozen to exactly zero
    dof_treeid = m.dof_treeid.numpy()
    asleep_dofs = d_ref.tree_awake.numpy()[:, dof_treeid] == 0
    self.assertTrue(asleep_dofs.any())
    self.assertTrue((d_fused.qacc_smooth.numpy()[asleep_dofs] == 0.0).all())
    self._assert_solve_accuracy(
      "qacc_smooth", m, d_ref, d_fused, d_ref.qfrc_smooth.numpy(), d_ref.qacc_smooth.numpy(), d_fused.qacc_smooth.numpy()
    )

    # the constraint solver ran on identical inputs
    self.assertTrue(np.isfinite(d_fused.qacc.numpy()).all())
    self._assert_close("qfrc_constraint", d_fused.qfrc_constraint.numpy(), d_ref.qfrc_constraint.numpy(), rtol=1e-3, atol=1e-3)

  def test_constraints_match_stock(self):
    """Condim 1/3/4/6 contacts, dof friction, limits, JOINT equalities: rows/islands/sleep agree."""
    mjm, m, datas = self._make(seed=4, contact=True, equality=True, resting=True)
    d_ref, d_fused = datas
    # world 6: the two-joint equality is inactive; world 7: both equalities inactive
    eq_active = d_ref.eq_active.numpy()
    eq_active[6, 0] = False
    eq_active[7, :] = False
    # world 2: a resting free body is asleep and stays asleep (its floor contacts still build rows);
    # world 3: the equality-coupled arm joints are asleep... the arm never sleeps (one tree), so
    # instead couple two free bodies via sleep and let the contact rows keep them in one island
    tree_asleep = d_ref.tree_asleep.numpy()
    tree_dofadr, tree_dofnum = m.tree_dofadr.numpy(), m.tree_dofnum.numpy()
    qvel = d_ref.qvel.numpy()
    for w, t in ((2, 5), (3, 7), (3, 8)):
      tree_asleep[w, t] = t
      qvel[w, tree_dofadr[t] : tree_dofadr[t] + tree_dofnum[t]] = 0.0
    for d in datas:
      d.eq_active.assign(eq_active)
      d.tree_asleep.assign(tree_asleep)
      d.qvel.assign(qvel)
      sleep.update_sleep(m, d)
    _share_contacts(m, d_ref, d_fused)
    self.assertGreater(int(d_ref.nacon.numpy()[0]), 4 * self.NWORLD)
    condims = d_ref.contact.dim.numpy()[: int(d_ref.nacon.numpy()[0])]
    self.assertEqual(set(condims.tolist()), {1, 3, 4, 6})

    self._run_forward(m, d_ref, fused=False)
    self._run_forward(m, d_fused, fused=True)

    self._assert_int_equal(
      d_fused,
      d_ref,
      (
        "tree_asleep",
        "tree_awake",
        "body_awake",
        "ntree_awake",
        "nbody_awake",
        "nv_awake",
        "nisland",
        "tree_island",
        "island_nv",
      ),
    )
    self._assert_int_equal(d_fused, d_ref, ("ncdof", "nsingleton6", "dof_cdof", "cdof_dof", "overflow"))
    self.assertEqual(int(d_ref.overflow.numpy().max()), 0)
    self._assert_awake_sets_equal(d_fused, d_ref)
    max_diff = self._assert_rows_equal(m, d_fused, d_ref)
    print("constraint row max abs diffs:", max_diff)

    # every row family is present and the islands are non-trivial
    types = d_ref.efc.type.numpy()
    nefc = d_ref.nefc.numpy()
    present = set()
    for w in range(self.NWORLD):
      present |= set(types[w, : nefc[w]].tolist())
    self.assertEqual(
      present,
      {
        ConstraintType.EQUALITY,
        ConstraintType.FRICTION_DOF,
        ConstraintType.LIMIT_JOINT,
        ConstraintType.CONTACT_FRICTIONLESS,
        ConstraintType.CONTACT_PYRAMIDAL,
      },
    )
    self.assertGreater(d_ref.ne.numpy()[0], 0)
    self.assertEqual(d_ref.ne.numpy()[7], 0)
    self.assertGreater(d_ref.nisland.numpy().min(), 1)
    self.assertEqual(d_ref.tree_awake.numpy()[2, 5], 0)

    # downstream: the compact solver ran on the same rows
    self._assert_close("qfrc_smooth", d_fused.qfrc_smooth.numpy(), d_ref.qfrc_smooth.numpy())
    self.assertTrue(np.isfinite(d_fused.qacc.numpy()).all())
    # the Newton solver stops at its cost tolerance: qacc of the 50 g free bodies carries the
    # constraint-force noise divided by their mass (the stock A/A envelope is of the same size)
    self._assert_close_scaled("qfrc_constraint", d_fused.qfrc_constraint.numpy(), d_ref.qfrc_constraint.numpy(), rel=1e-4)
    self._assert_close_scaled("qacc", d_fused.qacc.numpy(), d_ref.qacc.numpy(), rel=1e-4)

  def test_efc_overflow_matches_stock(self):
    """Rows beyond njmax are dropped and flagged like stock; the written rows are valid rows."""
    mjm, m, datas = self._make(seed=5, contact=True, equality=True, resting=True, nconmax=128, njmax=24, njmax_nnz=3072)
    d_ref, d_fused = datas
    for d in datas:
      sleep.update_sleep(m, d)
    _share_contacts(m, d_ref, d_fused)
    self._run_step(m, d_ref, fused=False)
    self._run_step(m, d_fused, fused=True)

    nefc = d_ref.nefc.numpy()
    self.assertGreater(nefc.min(), d_ref.njmax)
    # which contact rows land below njmax follows the allocation order (stock: atomics, fused:
    # bucket order), so the islands built from them are not comparable in the overflow case
    self._assert_int_equal(d_fused, d_ref, ("ne", "nf", "nl", "nefc", "overflow", "tree_asleep", "tree_awake"))
    overflow = d_ref.overflow.numpy()
    self.assertTrue((overflow & OverflowType.NEFC).all())
    self.assertFalse((overflow & OverflowType.NJMAX_NNZ).any())
    # non-contact families precede the contacts, so their rows are identical; the contact rows that
    # fit below njmax depend on the allocation order (stock: atomics) and are compared as a subset
    # of the rows a large-capacity run produces
    _, m_full, datas_full = self._make(
      seed=5, contact=True, equality=True, resting=True, nconmax=128, njmax=512, njmax_nnz=8192
    )
    d_full = datas_full[0]
    self.assertGreaterEqual(d_full.njmax, nefc.max())
    sleep.update_sleep(m_full, d_full)
    for f in dataclasses.fields(d_ref.contact):
      src = getattr(d_ref.contact, f.name)
      if isinstance(src, wp.array):
        wp.copy(getattr(d_full.contact, f.name), src)
    wp.copy(d_full.nacon, d_ref.nacon)
    self._run_forward(m_full, d_full, fused=False)
    contact_types = (ConstraintType.CONTACT_FRICTIONLESS, ConstraintType.CONTACT_PYRAMIDAL)
    for w in range(self.NWORLD):
      rows_full = _rows(m_full, d_full, w)
      for d in datas:
        rows = _rows(m, d, w)
        self.assertEqual(len(rows), d.njmax)
        for key, (scal, jac, _) in rows.items():
          scal_full, jac_full, _ = rows_full[key]
          for name in scal:
            np.testing.assert_allclose(scal[name], scal_full[name], rtol=_RTOL, atol=_ATOL, err_msg=f"row {key} {name}")
          self.assertEqual(sorted(jac), sorted(jac_full))
      self.assertEqual(
        {k for k in _rows(m, d_fused, w) if k[0] not in contact_types},
        {k for k in _rows(m, d_ref, w) if k[0] not in contact_types},
      )
      self.assertGreater(len([k for k in _rows(m, d_fused, w) if k[0] not in contact_types]), 0)
    self.assertTrue(np.isfinite(d_fused.qacc.numpy()).all())
    self.assertTrue(np.isfinite(d_fused.qpos.numpy()).all())

  def test_efc_nnz_overflow_matches_stock(self):
    """Jacobian nonzeros beyond njmax_nnz raise the NJMAX_NNZ flag on both paths."""
    mjm, m, datas = self._make(seed=5, contact=True, equality=True, resting=True, nconmax=128, njmax=512, njmax_nnz=96)
    d_ref, d_fused = datas
    for d in datas:
      sleep.update_sleep(m, d)
    _share_contacts(m, d_ref, d_fused)
    self._run_step(m, d_ref, fused=False)
    self._run_step(m, d_fused, fused=True)
    self._assert_int_equal(d_fused, d_ref, ("ne", "nf", "nl", "nefc", "overflow", "tree_asleep", "tree_awake"))
    overflow = d_ref.overflow.numpy()
    self.assertTrue((overflow & OverflowType.NJMAX_NNZ).all())
    self.assertFalse((overflow & OverflowType.NEFC).any())
    self.assertTrue(np.isfinite(d_fused.qacc.numpy()).all())
    self.assertTrue(np.isfinite(d_fused.qpos.numpy()).all())

  def test_contact_bucket_capacity(self):
    """Bucket capacity is exact when the budget allows; a saturated bucket flags NARROWPHASE."""
    self.assertEqual(fused_world.bucket_capacity(8, 1024), 1024)
    self.assertEqual(fused_world.bucket_capacity(1024, 1024 * 2400), (fused_world._GROUP_BUDGET_BYTES // 4) // 1024)
    self.assertEqual(fused_world.bucket_capacity(1 << 20, 1 << 20), (fused_world._GROUP_BUDGET_BYTES // 4) >> 20)
    mjm, m, datas = self._make(seed=8, contact=True, resting=True)
    d_ref, d_fused = datas
    for d in datas:
      sleep.update_sleep(m, d)
    _share_contacts(m, d_ref, d_fused)
    self._run_forward(m, d_ref, fused=False)
    # forward_m with buckets that hold two contacts per world
    fused_world.forward_a(m, d_fused)
    groups = fused_world.ContactGroups(2, wp.zeros((self.NWORLD,), dtype=int), wp.empty((2 * self.NWORLD,), dtype=int))
    fused_world._bucket_contacts(m, d_fused, groups)
    fused_world._launch_forward_m(m, d_fused, groups)
    wp.synchronize()
    self.assertTrue((d_fused.overflow.numpy() & OverflowType.NARROWPHASE).all())
    self.assertEqual(int(d_ref.overflow.numpy().max()), 0)
    self.assertTrue((groups.count.numpy() > 2).all())
    self.assertTrue((d_fused.nefc.numpy() < d_ref.nefc.numpy()).all())
    # the rows that were built are valid rows of the reference
    for w in range(self.NWORLD):
      rows_ref = _rows(m, d_ref, w)
      for key, (scal, jac, _) in _rows(m, d_fused, w).items():
        scal_ref, jac_ref, _ = rows_ref[key]
        for name in scal:
          np.testing.assert_allclose(scal[name], scal_ref[name], rtol=_RTOL, atol=_ATOL, err_msg=f"row {key} {name}")
        self.assertEqual(sorted(jac), sorted(jac_ref))

  def test_integration_matches_stock(self):
    """One step with contacts: implicitfast integration, sleep bookkeeping and the refresh agree."""
    for finalize in (True, False):
      mjm, m, datas = self._make(seed=6, contact=True, equality=True, resting=True)
      d_ref, d_fused = datas
      tree_asleep = d_ref.tree_asleep.numpy()
      tree_dofadr, tree_dofnum = m.tree_dofadr.numpy(), m.tree_dofnum.numpy()
      qvel = d_ref.qvel.numpy()
      # world 1: a resting body one substep away from sleeping; world 2: an asleep body; world 3:
      # trees 7 and 8 asleep in a cycle
      tree_asleep[1, 6] = -2
      qvel[1, tree_dofadr[6] : tree_dofadr[6] + tree_dofnum[6]] = 0.0
      tree_asleep[2, 5] = 5
      qvel[2, tree_dofadr[5] : tree_dofadr[5] + tree_dofnum[5]] = 0.0
      tree_asleep[3, 7] = 8
      tree_asleep[3, 8] = 7
      for t in (7, 8):
        qvel[3, tree_dofadr[t] : tree_dofadr[t] + tree_dofnum[t]] = 0.0
      for d in datas:
        d.tree_asleep.assign(tree_asleep)
        d.qvel.assign(qvel)
        sleep.update_sleep(m, d)
      _share_contacts(m, d_ref, d_fused)
      self._run_step(m, d_ref, fused=False, finalize=finalize)
      self._run_step(m, d_fused, fused=True, finalize=finalize)

      self._assert_int_equal(
        d_fused, d_ref, ("tree_asleep", "tree_awake", "body_awake", "ntree_awake", "nbody_awake", "nv_awake", "overflow")
      )
      self._assert_awake_sets_equal(d_fused, d_ref)
      self.assertEqual(d_ref.tree_awake.numpy()[2, 5], 0)
      self._assert_close("time", d_fused.time.numpy(), d_ref.time.numpy())
      # solver-tolerance limited (see test_constraints_match_stock)
      self._assert_close_scaled("qacc_warmstart", d_fused.qacc_warmstart.numpy(), d_ref.qacc_warmstart.numpy(), rel=1e-4)
      self._assert_close_scaled("qacc", d_fused.qacc.numpy(), d_ref.qacc.numpy(), rel=1e-4)
      # sleeping trees have exactly zero velocity and acceleration
      dof_treeid = m.dof_treeid.numpy()
      asleep = d_ref.tree_awake.numpy()[:, dof_treeid] == 0
      self.assertTrue(asleep.any())
      self.assertTrue((d_fused.qvel.numpy()[asleep] == 0.0).all())
      self.assertTrue((d_fused.qacc.numpy()[asleep] == 0.0).all())
      max_rel = {}
      for name in ("qvel", "qpos"):
        max_rel[name] = self._assert_close_scaled(name, getattr(d_fused, name).numpy(), getattr(d_ref, name).numpy(), rel=1e-4)
      # velocity-derived data: refreshed on the post-sleep qvel when finalize, otherwise left at the
      # values of the forward pass (both paths)
      refresh = (
        "actuator_velocity",
        "cvel",
        "cdof_dot",
        "qfrc_spring",
        "qfrc_damper",
        "qfrc_passive",
        "cacc",
        "cfrc_int",
        "qfrc_bias",
      )
      for name in refresh:
        max_rel[name] = self._assert_close_scaled(name, getattr(d_fused, name).numpy(), getattr(d_ref, name).numpy(), rel=1e-4)
      if finalize:
        # the refresh used the post-sleep velocities: sleeping trees have zero damper forces
        self.assertTrue((d_fused.qfrc_damper.numpy()[asleep] == 0.0).all())
        # springs were refreshed on the advanced qpos (the arm's j2 spring moves every substep)
        self.assertGreater(np.abs(d_fused.qfrc_spring.numpy()[:, 1]).max(), 0.0)
      print(f"integration finalize={finalize} max rel diffs: {max_rel}")

  def test_step_trajectory_matches_stock(self):
    """16 substeps: identical sleep-state sequence and matching qpos/qvel, fused vs stock."""
    mjm, m, datas = self._make(seed=2, qvel_scale=0.2)
    d_ref, d_fused = datas
    # hovering (gravcomp) free bodies start at rest so that they fall asleep through the countdown
    body_treeid = m.body_treeid.numpy()
    hover_trees = [
      int(body_treeid[b])
      for b in range(m.nbody)
      if mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_BODY, b).startswith("free") and mjm.body_gravcomp[b] > 0
    ]
    self.assertEqual(len(hover_trees), 3)
    tree_dofadr = m.tree_dofadr.numpy()
    tree_dofnum = m.tree_dofnum.numpy()
    qvel = d_ref.qvel.numpy()
    for t in hover_trees:
      qvel[:, tree_dofadr[t] : tree_dofadr[t] + tree_dofnum[t]] = 0.0
    # one free body per world already asleep
    tree_asleep = d_ref.tree_asleep.numpy()
    tree_asleep[:, 10] = 10
    qvel[:, tree_dofadr[10] : tree_dofadr[10] + tree_dofnum[10]] = 0.0
    ctrl = d_ref.ctrl.numpy()
    ctrl[:, 7:9] = 0.02  # finger targets inside the range
    for d in datas:
      d.qvel.assign(qvel)
      d.tree_asleep.assign(tree_asleep)
      d.ctrl.assign(ctrl)
      sleep.update_sleep(m, d)

    max_rel = {"qpos": 0.0, "qvel": 0.0}
    asleep_seq = []
    for step in range(16):
      self._run_step(m, d_ref, fused=False)
      self._run_step(m, d_fused, fused=True)
      np.testing.assert_array_equal(d_fused.tree_asleep.numpy(), d_ref.tree_asleep.numpy(), err_msg=f"tree_asleep step {step}")
      np.testing.assert_array_equal(d_fused.tree_awake.numpy(), d_ref.tree_awake.numpy(), err_msg=f"tree_awake step {step}")
      asleep_seq.append(d_ref.tree_awake.numpy().sum(axis=1).copy())
      for name in max_rel:
        ref = getattr(d_ref, name).numpy()
        fused = getattr(d_fused, name).numpy()
        self.assertTrue(np.isfinite(fused).all(), f"{name} not finite at step {step}")
        scale = max(float(np.abs(ref).max()), 1e-6)
        max_rel[name] = max(max_rel[name], float(np.abs(fused - ref).max()) / scale)
    print("trajectory (no contacts) max rel diffs:", max_rel)
    for name, value in max_rel.items():
      self.assertLessEqual(value, 1e-4, f"{name} max rel diff {value}")

    # the hovering bodies fell asleep along the way and the pre-asleep tree never woke
    awake_counts = np.stack(asleep_seq)
    self.assertLess(awake_counts[-1].max(), awake_counts[0].min())
    self.assertEqual(d_ref.tree_awake.numpy()[:, 10].max(), 0)
    tree_awake = d_ref.tree_awake.numpy()
    for t in hover_trees:
      self.assertEqual(tree_awake[:, t].max(), 0, f"hover tree {t} still awake")

  def test_contact_trajectory_matches_stock(self):
    """32 substeps with external floor contacts: resting bodies fall asleep identically."""
    mjm, m, datas = self._make(seed=7, contact=True, equality=True, resting=True)
    d_ref, d_fused = datas
    qvel = d_ref.qvel.numpy()
    qvel[:, 9:] = 0.0  # free bodies start at rest on the floor
    for d in datas:
      d.qvel.assign(qvel)
      sleep.update_sleep(m, d)

    max_rel = {"qpos": 0.0, "qvel": 0.0}
    awake_seq = []
    for step in range(32):
      # the contact set is refreshed every substep from the reference state, like Newton does
      _share_contacts(m, d_ref, d_fused)
      self._run_step(m, d_ref, fused=False, finalize=(step % 8 == 7))
      self._run_step(m, d_fused, fused=True, finalize=(step % 8 == 7))
      np.testing.assert_array_equal(d_fused.tree_asleep.numpy(), d_ref.tree_asleep.numpy(), err_msg=f"tree_asleep step {step}")
      np.testing.assert_array_equal(d_fused.tree_island.numpy(), d_ref.tree_island.numpy(), err_msg=f"tree_island step {step}")
      np.testing.assert_array_equal(d_fused.nefc.numpy(), d_ref.nefc.numpy(), err_msg=f"nefc step {step}")
      awake_seq.append(d_ref.tree_awake.numpy().sum(axis=1).copy())
      for name in max_rel:
        ref = getattr(d_ref, name).numpy()
        fused = getattr(d_fused, name).numpy()
        self.assertTrue(np.isfinite(fused).all(), f"{name} not finite at step {step}")
        scale = max(float(np.abs(ref).max()), 1e-6)
        max_rel[name] = max(max_rel[name], float(np.abs(fused - ref).max()) / scale)
    print("trajectory (contacts) max rel diffs:", max_rel)
    for name, value in max_rel.items():
      self.assertLessEqual(value, 1e-4, f"{name} max rel diff {value}")
    awake_counts = np.stack(awake_seq)
    self.assertLess(awake_counts[-1].max(), awake_counts[0].min())
    self.assertGreater(d_ref.nefc.numpy().min(), 0)


if __name__ == "__main__":
  wp.init()
  absltest.main()
