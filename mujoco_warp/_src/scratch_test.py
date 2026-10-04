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
"""Stages that declare their own scratch: eager, discovered and prepared paths agree."""

import dataclasses

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized
from gpu_components import scratch as scratch_ops

import mujoco_warp as mjw
from mujoco_warp import test_data
from mujoco_warp._src.types import DisableBit
from mujoco_warp._src.types import IntegratorType


def _pendula(integrator, disableflags):
  mjm, mjd, _, _ = test_data.fixture(
    "pendula.xml",
    overrides={
      "opt.disableflags": disableflags,
      "opt.integrator": integrator,
      "opt.enableflags": mjw.EnableBit.INVDISCRETE,
    },
  )
  np.random.seed(0)
  mjd.qvel = np.random.uniform(low=-0.01, high=0.01, size=mjd.qvel.shape)
  mjd.ctrl = np.random.uniform(low=-0.1, high=0.1, size=mjd.ctrl.shape)
  mujoco.mj_forward(mjm, mjd)
  return mjm, mjd


class ScratchStagesTest(parameterized.TestCase):
  @parameterized.named_parameters(
    ("implicit", IntegratorType.IMPLICITFAST, DisableBit.CONTACT, 6),
    ("euler_damping", IntegratorType.EULER, DisableBit.CONTACT, 5),  # pinned: EULERDAMP and DAMPER enabled
  )
  def test_eager_discovered_and_prepared_paths_agree(self, integrator, disableflags, declared):
    mjm, mjd = _pendula(integrator, disableflags)
    m = mjw.put_model(mjm)
    stage = mjw.implicit if integrator == IntegratorType.IMPLICITFAST else mjw.euler

    def fresh(registry):
      d = mjw.put_data(mjm, mjd)
      d.efc.Ma = wp.array(mjd.qfrc_constraint + mjd.qfrc_smooth, dtype=wp.float32, shape=(1, -1))
      return dataclasses.replace(d, scratch=registry)

    eager = fresh(None)
    stage(m, eager)

    registry = scratch_ops.allocate_registry(eager.qpos.device)
    discovered = fresh(registry)
    stage(m, discovered)  # discovery: plain arrays, requests recorded by name
    self.assertEqual(len(registry.declarations), declared)
    self.assertTrue(all(name.startswith(("implicit.", "euler.")) for name in registry.declarations))
    np.testing.assert_array_equal(discovered.qpos.numpy(), eager.qpos.numpy())

    scratch_ops.freeze(registry)
    prepared = fresh(registry)
    with self.assertRaisesRegex(RuntimeError, "frozen"):
      stage(m, prepared)  # G5(b): a frozen registry answers nothing
    columns = {
      name: wp.zeros(scratch_ops.bounded_shape(decl.shape), dtype=decl.dtype, device=eager.qpos.device)
      for name, decl in registry.declarations.items()
    }
    scratch_ops.prepare(registry, columns)
    stage(m, prepared)
    np.testing.assert_array_equal(prepared.qpos.numpy(), eager.qpos.numpy())
    np.testing.assert_array_equal(prepared.qvel.numpy(), eager.qvel.numpy())

    # G7: a reused column is initialized by the stage every execution, not by its allocation
    for column in columns.values():
      column.fill_(float("nan") if column.dtype == wp.float32 else -7)
    poisoned = fresh(registry)
    stage(m, poisoned)
    np.testing.assert_array_equal(poisoned.qpos.numpy(), eager.qpos.numpy())
    np.testing.assert_array_equal(poisoned.qvel.numpy(), eager.qvel.numpy())

  def test_fixed_execution_rejects_a_registry_and_dynamic_count_needs_one(self):
    with self.assertRaisesRegex(RuntimeError, "requires a prepared registry"):
      mjw.scratch_array(None, "x", (wp.CountParameter(4), 2), float)


if __name__ == "__main__":
  wp.init()
  absltest.main()
