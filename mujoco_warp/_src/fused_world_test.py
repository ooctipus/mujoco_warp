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

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest

import mujoco_warp as mjw
from mujoco_warp import test_data
from mujoco_warp._src import fused_world
from mujoco_warp._src import sleep
from mujoco_warp._src import smooth

# fused and stock paths differ only in floating point summation / factorization order
_RTOL = 1e-5
_ATOL = 1e-6

_OPTION = """
  <option integrator="implicitfast" cone="pyramidal" solver="Newton" jacobian="sparse" sleep_tolerance="0.01"
          gravity="0 0 -9.81">
    <flag sleep="enable" island="enable" contact="disable"/>
  </option>
"""


def factory_like_xml(n_free: int = 19, n_fixed: int = 19, n_hover: int = 3) -> str:
  """Franka-like 9-DOF chain (11 bodies) with servos, limits and gravcomp, plus free and static bodies.

  ``n_hover`` of the free bodies carry gravcomp=1 so that they hover with zero velocity and fall
  asleep through the MJ_MINAWAKE countdown during a trajectory.
  """
  chain = """
    <body name="link0" pos="0 0 0">
      <geom type="box" size="0.05 0.05 0.05" mass="3"/>
      <body name="link1" pos="0 0 0.15" gravcomp="1"><joint name="j1" type="hinge" axis="0 0 1" range="-2.9 2.9" damping="0.5" armature="0.1"/><geom type="capsule" size="0.04" fromto="0 0 0 0 0 0.2" mass="3"/>
      <body name="link2" pos="0 0 0.2" gravcomp="1"><joint name="j2" type="hinge" axis="0 1 0" range="-1.7 1.7" damping="0.5" armature="0.1" stiffness="2"/><geom type="capsule" size="0.04" fromto="0 0 0 0 0 0.2" mass="3"/>
      <body name="link3" pos="0 0 0.2" gravcomp="1"><joint name="j3" type="hinge" axis="0 0 1" range="-2.9 2.9" damping="0.3" armature="0.05"/><geom type="capsule" size="0.04" fromto="0 0 0 0 0 0.2" mass="2.5"/>
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
  simple_geom = '<geom type="box" size="0.02 0.02 0.02" mass="0.05"/>'
  offset_geom = '<geom type="box" size="0.02 0.03 0.04" pos="0.01 0.005 0.0" euler="0.3 0.2 0.1" mass="0.05"/>'
  free = "".join(
    f'<body name="free{i}" pos="{0.5 + 0.1 * i} 0 0.3" gravcomp="{1 if i < n_hover else 0}">'
    f"<freejoint/>{simple_geom if i % 2 == 0 else offset_geom}</body>\n"
    for i in range(n_free)
  )
  fixed = "".join(
    f'<body name="fixed{i}" pos="{0.5 + 0.1 * i} 0.3 0"><geom type="box" size="0.03 0.03 0.01"/></body>\n' for i in range(n_fixed)
  )
  actuators = "".join(
    f'<position joint="j{i}" kp="{50.0 + 10.0 * i}" kv="{2.0 + i}" ctrlrange="-2.5 2.5" forcerange="-80 80"/>\n' for i in range(1, 8)
  )
  actuators += '<position joint="fl" kp="200" ctrlrange="0 0.04"/>\n<position joint="fr" kp="200" ctrlrange="0 0.04"/>\n'
  actuators += "".join(f'<velocity joint="j{i}" kv="{0.5 * i}" ctrlrange="-1 1"/>\n' for i in range(1, 8))
  return f"""
  <mujoco>
    {_OPTION}
    <worldbody>
      {chain}
      {free}
      {fixed}
    </worldbody>
    <actuator>
      {actuators}
    </actuator>
  </mujoco>
  """


def _random_state(mjm: mujoco.MjModel, nworld: int, rng: np.random.Generator, qvel_scale: float = 1.0):
  """Per-world random qpos (unit quaternions for free joints), qvel and ctrl."""
  qpos = np.tile(mjm.qpos0, (nworld, 1))
  qvel = rng.uniform(-qvel_scale, qvel_scale, size=(nworld, mjm.nv))
  for jntid in range(mjm.njnt):
    adr = mjm.jnt_qposadr[jntid]
    if mjm.jnt_type[jntid] == mujoco.mjtJoint.mjJNT_FREE:
      qpos[:, adr : adr + 3] += rng.uniform(-0.2, 0.2, size=(nworld, 3))
      quat = rng.normal(size=(nworld, 4))
      qpos[:, adr + 3 : adr + 7] = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    else:
      lo, hi = mjm.jnt_range[jntid]
      # some worlds sit outside the limits so that limit rows are active
      qpos[:, adr] = rng.uniform(lo - 0.2, hi + 0.2, size=nworld)
  ctrl = rng.uniform(-1.0, 1.0, size=(nworld, mjm.nu))
  return qpos.astype(np.float32), qvel.astype(np.float32), ctrl.astype(np.float32)


class FusedWorldTest(absltest.TestCase):
  NWORLD = 8

  def _make(self, seed: int, qvel_scale: float = 1.0):
    mjm, mjd, m, _ = test_data.fixture(xml=factory_like_xml(), nworld=1)
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
    qpos, qvel, ctrl = _random_state(mjm, self.NWORLD, rng, qvel_scale)
    qfrc_applied = np.zeros((self.NWORLD, mjm.nv), dtype=np.float32)
    xfrc_applied = np.zeros((self.NWORLD, mjm.nbody, 6), dtype=np.float32)
    # applied forces in a few worlds: generalized on the arm, Cartesian on a free body and the hand
    qfrc_applied[1, :7] = rng.uniform(-1.0, 1.0, size=7)
    xfrc_applied[1, 16] = rng.uniform(-1.0, 1.0, size=6)
    xfrc_applied[6, 9] = rng.uniform(-2.0, 2.0, size=6)

    datas = []
    for _ in range(2):
      d = mjw.put_data(mjm, mjd, nworld=self.NWORLD)
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
    fused_world.enabled = fused
    try:
      if fused:
        self.assertTrue(fused_world.fused_world(m, d))
      mjw.forward(m, d)
      wp.synchronize()
    finally:
      fused_world.enabled = True

  def _assert_close(self, name, fused, ref, rtol=_RTOL, atol=_ATOL):
    fused = np.asarray(fused)
    ref = np.asarray(ref)
    self.assertTrue(np.isfinite(fused).all(), f"{name} not finite")
    np.testing.assert_allclose(fused, ref, rtol=rtol, atol=atol, err_msg=f"mismatch: {name}")

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
    self.assertLessEqual(err_fused, max(2.0 * err_ref, 1e-6), f"{name}: fused solve less accurate than stock")
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
    fused_world.enabled = False
    try:
      self.assertFalse(fused_world.fused_world(m, d))
    finally:
      fused_world.enabled = True

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
    for name in exact_int:
      np.testing.assert_array_equal(getattr(d_fused, name).numpy(), getattr(d_ref, name).numpy(), err_msg=f"mismatch: {name}")

    # a few worlds must actually be asleep / woken for the comparison to be meaningful
    tree_awake = d_fused.tree_awake.numpy()
    self.assertEqual(tree_awake[2, 1], 0)
    self.assertEqual(tree_awake[2, 2], 0)
    self.assertEqual(tree_awake[3, 3], 0)
    self.assertEqual(tree_awake[3, 4], 0)
    self.assertEqual(tree_awake[4, 5], 1)
    self.assertEqual(tree_awake[5, 6], 1)
    self.assertLess(d_fused.tree_asleep.numpy()[4, 5], 0)

    # awake index compaction: the native update_sleep after make_constraint rewrites these in
    # atomic order, so compare as sets
    for name, count in (("body_awake_ind", "nbody_awake"), ("dof_awake_ind", "nv_awake")):
      n = getattr(d_ref, count).numpy()
      for w in range(self.NWORLD):
        ref_set = set(getattr(d_ref, name).numpy()[w, : n[w]].tolist())
        fused_ind = getattr(d_fused, name).numpy()[w, : n[w]]
        self.assertEqual(set(fused_ind.tolist()), ref_set, f"mismatch: {name} world {w}")

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

    # gravcomp and limits are exercised
    self.assertGreater(np.abs(d_ref.qfrc_gravcomp.numpy()).max(), 1.0)
    self.assertGreater(d_ref.nl.numpy().max(), 0)
    self.assertGreater(np.abs(d_ref.qfrc_applied.numpy()).max(), 0.0)

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
    # qacc_smooth: the 9-dof block uses tile Cholesky in stock and scalar Cholesky in the fused path;
    # dofs of sleeping trees are frozen to exactly zero
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

  def test_step_trajectory_matches_stock(self):
    """16 substeps: identical sleep-state sequence and matching qpos/qvel between fused and stock."""
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
      fused_world.enabled = False
      mjw.step(m, d_ref)
      fused_world.enabled = True
      mjw.step(m, d_fused)
      wp.synchronize()
      np.testing.assert_array_equal(d_fused.tree_asleep.numpy(), d_ref.tree_asleep.numpy(), err_msg=f"tree_asleep step {step}")
      np.testing.assert_array_equal(d_fused.tree_awake.numpy(), d_ref.tree_awake.numpy(), err_msg=f"tree_awake step {step}")
      asleep_seq.append(d_ref.tree_awake.numpy().sum(axis=1).copy())
      for name in max_rel:
        ref = getattr(d_ref, name).numpy()
        fused = getattr(d_fused, name).numpy()
        self.assertTrue(np.isfinite(fused).all(), f"{name} not finite at step {step}")
        scale = max(float(np.abs(ref).max()), 1e-6)
        max_rel[name] = max(max_rel[name], float(np.abs(fused - ref).max()) / scale)
    for name, value in max_rel.items():
      self.assertLessEqual(value, 1e-4, f"{name} max rel diff {value}")

    # the hovering bodies fell asleep along the way and the pre-asleep tree never woke
    awake_counts = np.stack(asleep_seq)
    self.assertLess(awake_counts[-1].max(), awake_counts[0].min())
    self.assertEqual(d_ref.tree_awake.numpy()[:, 10].max(), 0)
    tree_awake = d_ref.tree_awake.numpy()
    for t in hover_trees:
      self.assertEqual(tree_awake[:, t].max(), 0, f"hover tree {t} still awake")


if __name__ == "__main__":
  wp.init()
  absltest.main()
