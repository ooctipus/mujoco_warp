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

"""Tests for the fused single-launch tree traversal used by com_pos, crb and rne."""

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest

import mujoco_warp as mjw
from mujoco_warp import DisableBit
from mujoco_warp import test_data

# fused and per-level paths differ only in float atomic summation order
_RTOL = 1e-6
_ATOL = 1e-7

# a serial chain with one branch point, several free bodies and static bodies
_XML = """
<mujoco>
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <body name="base" pos="0 0 0.5">
      <geom type="box" size="0.05 0.05 0.05" mass="1"/>
      <body name="l1" pos="0 0 0.1">
        <joint type="hinge" axis="0 1 0"/>
        <geom type="capsule" size="0.02" fromto="0 0 0 0 0 0.2" mass="0.5"/>
        <body name="l2" pos="0 0 0.2">
          <joint type="hinge" axis="1 0 0"/>
          <geom type="capsule" size="0.02" fromto="0 0 0 0 0 0.2" mass="0.4"/>
          <body name="l3" pos="0 0 0.2">
            <joint type="hinge" axis="0 1 0"/>
            <geom type="capsule" size="0.02" fromto="0 0 0 0 0 0.2" mass="0.3"/>
            <body name="l4" pos="0 0 0.2">
              <joint type="slide" axis="0 0 1"/>
              <geom type="capsule" size="0.02" fromto="0 0 0 0 0 0.15" mass="0.2"/>
              <body name="finger_l" pos="0.03 0 0.15">
                <joint type="slide" axis="1 0 0"/>
                <geom type="box" size="0.01 0.01 0.03" mass="0.05"/>
              </body>
              <body name="finger_r" pos="-0.03 0 0.15">
                <joint type="slide" axis="-1 0 0"/>
                <geom type="box" size="0.01 0.01 0.03" mass="0.05"/>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
    <body name="free0" pos="0.5 0 0.3"><freejoint/><geom type="box" size="0.02 0.03 0.04" mass="0.1"/></body>
    <body name="free1" pos="0.6 0 0.3"><freejoint/><geom type="sphere" size="0.03" mass="0.2"/></body>
    <body name="free2" pos="0.7 0 0.3"><freejoint/><geom type="capsule" size="0.01 0.05" mass="0.05"/></body>
    <body name="fixed0" pos="0.5 0.3 0"><geom type="box" size="0.05 0.05 0.02"/></body>
    <body name="fixed1" pos="0.6 0.3 0"><geom type="box" size="0.05 0.05 0.02"/></body>
  </worldbody>
</mujoco>
"""


def _random_state(mjm: mujoco.MjModel, nworld: int, rng: np.random.Generator):
  """Per-world random qpos (unit quaternions for free joints) and qvel."""
  qpos = np.tile(mjm.qpos0, (nworld, 1))
  qvel = rng.uniform(-1.0, 1.0, size=(nworld, mjm.nv))
  for jntid in range(mjm.njnt):
    adr = mjm.jnt_qposadr[jntid]
    if mjm.jnt_type[jntid] == mujoco.mjtJoint.mjJNT_FREE:
      qpos[:, adr : adr + 3] += rng.uniform(-0.2, 0.2, size=(nworld, 3))
      quat = rng.normal(size=(nworld, 4))
      qpos[:, adr + 3 : adr + 7] = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    else:
      qpos[:, adr] += rng.uniform(-1.0, 1.0, size=nworld)
  return qpos, qvel


def _run_fwd(m: mjw.Model, d: mjw.Data, qpos: np.ndarray, qvel: np.ndarray, tree_block_dim: int):
  m.block_dim.tree_accumulate = tree_block_dim
  d.qpos.assign(qpos)
  d.qvel.assign(qvel)
  for arr in (d.subtree_com, d.crb, d.cfrc_int, d.qfrc_bias):
    arr.fill_(wp.inf)
  mjw.fwd_position(m, d)
  mjw.fwd_velocity(m, d)
  return {name: getattr(d, name).numpy().copy() for name in ("subtree_com", "crb", "M", "cfrc_int", "qfrc_bias")}


class SmoothFusedTest(absltest.TestCase):
  def test_level_table_matches_body_tree(self):
    """The flattened level table concatenates body_tree levels in order."""
    _, _, m, _ = test_data.fixture(xml=_XML)
    offsets = m.body_tree_offsets.numpy()
    flat = m.body_tree_all.numpy()
    self.assertEqual(len(offsets), len(m.body_tree) + 1)
    self.assertEqual(offsets[-1], m.nbody)
    for level, body_tree in enumerate(m.body_tree):
      np.testing.assert_array_equal(flat[offsets[level] : offsets[level + 1]], body_tree.numpy())
    # a non-trivial tree so the test exercises multiple levels and sibling accumulation
    self.assertGreaterEqual(len(m.body_tree), 7)

  def test_fused_matches_per_level(self):
    """Fused single-launch traversal reproduces the per-level launches across worlds."""
    nworld = 8
    mjm, mjd, m, d = test_data.fixture(xml=_XML, nworld=nworld, overrides={"opt.disableflags": DisableBit.CONTACT})
    m.opt.run_collision_detection = False
    if not d.qpos.device.is_cuda:
      self.skipTest("fused tree traversal only runs on CUDA")

    qpos, qvel = _random_state(mjm, nworld, np.random.default_rng(0))
    ref = _run_fwd(m, d, qpos, qvel, tree_block_dim=0)
    fused = _run_fwd(m, d, qpos, qvel, tree_block_dim=128)

    for name in ref:
      self.assertTrue(np.isfinite(fused[name]).all(), f"{name} not finite")
      np.testing.assert_allclose(fused[name], ref[name], rtol=_RTOL, atol=_ATOL, err_msg=f"mismatch: {name}")

    # anchor the fused path against MuJoCo for one world so that the two MJWarp paths are not merely
    # self-consistent
    mjd.qpos[:] = qpos[3]
    mjd.qvel[:] = qvel[3]
    mujoco.mj_forward(mjm, mjd)
    tol = 5e-4
    np.testing.assert_allclose(fused["subtree_com"][3], mjd.subtree_com, rtol=tol, atol=tol)
    np.testing.assert_allclose(fused["crb"][3], mjd.crb, rtol=tol, atol=tol)
    np.testing.assert_allclose(fused["M"][3], mjd.M, rtol=tol, atol=tol)
    np.testing.assert_allclose(fused["qfrc_bias"][3], mjd.qfrc_bias, rtol=tol, atol=tol)

  def test_fused_block_dim_smaller_than_level(self):
    """Levels wider than the block are covered by the strided loop inside the CTA."""
    nworld = 4
    mjm, _, m, d = test_data.fixture(xml=_XML, nworld=nworld, overrides={"opt.disableflags": DisableBit.CONTACT})
    m.opt.run_collision_detection = False
    if not d.qpos.device.is_cuda:
      self.skipTest("fused tree traversal only runs on CUDA")
    # widest level has 6 bodies (base, 3 free, 2 static): a 4-thread block needs a second stride
    self.assertGreater(int(np.diff(m.body_tree_offsets.numpy()).max()), 4)

    qpos, qvel = _random_state(mjm, nworld, np.random.default_rng(1))
    ref = _run_fwd(m, d, qpos, qvel, tree_block_dim=0)
    fused = _run_fwd(m, d, qpos, qvel, tree_block_dim=4)
    for name in ref:
      np.testing.assert_allclose(fused[name], ref[name], rtol=_RTOL, atol=_ATOL, err_msg=f"mismatch: {name}")


if __name__ == "__main__":
  wp.init()
  absltest.main()
