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

_IMPLICIT = {f"implicit.{name}" for name in ("qDeriv", "qLD", "qLDiagInv", "actuator_vel", "qacc", "island_can_sleep")}
_EULER = {f"euler.{name}" for name in ("qacc", "damp_deriv", "M", "qLD", "qLDiagInv")}
_FORWARD = {
  "position.awake_prev",
  "broadphase.awake_changed",
  "constraint.efc_nnz",
  "island.parent",
  "island.efc_tree",
  "transmission.moment_nnz",
  "solver.nsolving",
  "solver.grad",
  "solver.ls_exhausted",
  "collision.pair",
  "collision.pairid",
  "collision.worldid",
  "convex.nccd",
  "convex.epa_vert",
}
_SLEEPING_BOXES = """
<mujoco>
  <option solver="Newton" integrator="implicitfast" cone="pyramidal" jacobian="sparse">
    <flag sleep="enable" island="enable"/>
  </option>
  <worldbody>
    <geom type="plane" size="2 2 .1"/>
    <body pos="0 0 .3"><freejoint/><geom type="box" size=".1 .1 .1" mass="1"/></body>
    <body pos="1 0 .3"><freejoint/><geom type="box" size=".1 .1 .1" mass="1"/></body>
    <body pos="0 1 .05"><freejoint/><geom type="sphere" size=".1" mass="1"/></body>
  </worldbody>
</mujoco>
"""


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
    ("implicit", "pendula", IntegratorType.IMPLICITFAST, DisableBit.CONTACT, "implicit", _IMPLICIT, ("qpos", "qvel"), 0.0),
    ("euler_damping", "pendula", IntegratorType.EULER, DisableBit.CONTACT, "euler", _EULER, ("qpos", "qvel"), 0.0),
    # Sleeping, islands, NXN contacts and compact Newton: required fields span all scratch domains.
    (
      "forward_sleeping",
      "boxes",
      None,
      None,
      "forward",
      _FORWARD,
      ("qacc", "qfrc_constraint", "nefc", "nacon", "nisland"),
      1e-5,
    ),
  )
  def test_eager_discovered_and_prepared_paths_agree(
    self, fixture, integrator, disableflags, stage, declared, fields, tolerance
  ):
    if fixture == "pendula":
      mjm, mjd = _pendula(integrator, disableflags)
    else:
      mjm, mjd, _, _ = test_data.fixture(xml=_SLEEPING_BOXES)
    m = mjw.put_model(mjm)
    if fixture == "boxes":
      m.opt.broadphase = mjw.BroadphaseType.NXN
      m.opt.graph_conditional = True
    stage = getattr(mjw, stage)

    def fresh(registry):
      d = mjw.put_data(mjm, mjd, nworld=2 if fixture == "boxes" else 1)
      if fixture == "pendula":
        d.efc.Ma = wp.array(mjd.qfrc_constraint + mjd.qfrc_smooth, dtype=wp.float32, shape=(1, -1))
      return dataclasses.replace(d, scratch=registry)

    def check(d, reference):
      for name in fields:
        actual, expected = getattr(d, name).numpy(), getattr(reference, name).numpy()
        self.assertTrue(np.isfinite(actual).all() and np.isfinite(expected).all(), name)
        np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance, equal_nan=False)

    eager = fresh(None)
    stage(m, eager)

    registry = scratch_ops.allocate_registry(eager.qpos.device)
    discovered = fresh(registry)
    stage(m, discovered)  # discovery: plain arrays, requests recorded by name
    self.assertLessEqual(declared, set(registry.declarations))
    check(discovered, eager)

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
    check(prepared, eager)

    # G7: a reused column is initialized by the stage every execution, not by its allocation
    for column in columns.values():
      scalar_type = getattr(column.dtype, "_wp_scalar_type_", column.dtype)
      poison = float("nan") if scalar_type == wp.float32 else -7
      column.fill_(column.dtype(poison) if scalar_type != column.dtype else poison)
    poisoned = fresh(registry)
    stage(m, poisoned)
    check(poisoned, eager)

  def test_dynamic_count_needs_a_registry(self):
    with self.assertRaisesRegex(RuntimeError, "requires a prepared registry"):
      scratch_ops.scratch_array(None, "x", (wp.CountParameter(4), 2), float)


if __name__ == "__main__":
  wp.init()
  absltest.main()
