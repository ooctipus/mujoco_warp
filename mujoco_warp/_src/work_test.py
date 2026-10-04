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
"""Native work contracts, dependency boundaries and numerical reuse."""

import ast
import dataclasses
from pathlib import Path
from unittest import mock

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjw
from mujoco_warp import test_data
from mujoco_warp._src import forward
from mujoco_warp._src import solver
from mujoco_warp._src import types

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


class NativeWorkTest(parameterized.TestCase):
  def test_package_has_no_managed_runtime_dependency(self):
    root = Path(mjw.__file__).parent
    forbidden = {"StepBindings", "ScratchRegistry", "scratch_array"}
    for path in root.rglob("*.py"):
      if path.name.endswith("_test.py") or path.name == "conftest.py" or "test_data" in path.parts:
        continue
      for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom):
          self.assertFalse((node.module or "").startswith("gpu_components"), str(path))
          self.assertFalse({item.name for item in node.names} & forbidden, str(path))
        elif isinstance(node, ast.Import):
          self.assertFalse(any(item.name.startswith("gpu_components") for item in node.names), str(path))
        elif isinstance(node, (ast.Name, ast.Attribute)):
          self.assertNotIn(node.id if isinstance(node, ast.Name) else node.attr, forbidden, str(path))
    for name in ("workspace.py", "step_program.py", "step_execution.py"):
      self.assertFalse((root / "_src" / name).exists(), name)
    self.assertNotIn("scratch", {field.name for field in dataclasses.fields(types.Data)})
    self.assertNotIn("gpu-components", (root.parent / "pyproject.toml").read_text())
    self.assertNotIn("gpu-components", (root.parent / "uv.lock").read_text())

  def test_symbolic_layout_binding_and_count_identity(self):
    _, _, m, d = test_data.fixture(xml=_SLEEPING_BOXES, nworld=2)
    worlds, candidates, ccd = (wp.CountParameter(11) for _ in range(3))
    execution = dataclasses.replace(d, nworld=worlds, naconmax=candidates, naccdmax=ccd)
    layout = mjw.step_work_layout(m, execution)
    self.assertIs(layout["position.moment_nnz"][0][0], worlds)
    self.assertIs(layout["position.collision.collision_pair"][0][0], candidates)
    self.assertIs(layout["position.convex.epa_vert"][0][0], ccd)
    self.assertEqual(layout["solver.nsolving"][0], (1,))
    arrays = {name: wp.empty(tuple(wp.upper_bound(x) for x in shape), dtype=dtype) for name, (shape, dtype) in layout.items()}
    with mock.patch.object(wp, "empty", side_effect=AssertionError("Borrowed preparation allocated payload")):
      work = mjw.make_step_work(m, execution, arrays)
    self.assertEqual({name for name, _, _ in mjw.array_fields(work)}, layout.keys())
    for name, array, _ in mjw.array_fields(work):
      self.assertIs(array, arrays[name])
    incomplete = dict(arrays)
    incomplete.pop("solver.nsolving")
    with self.assertRaisesRegex(ValueError, "exactly"):
      mjw.make_step_work(m, execution, incomplete)
    wrong = dict(arrays, **{"solver.nsolving": wp.empty(2, dtype=int)})
    with self.assertRaisesRegex(ValueError, "incompatible"):
      mjw.make_step_work(m, execution, wrong)
    before = mjw.metadata_signature((m, d, execution, work))
    execution.nworld = wp.CountParameter(11)
    self.assertNotEqual(before, mjw.metadata_signature((m, d, execution, work)))

  def test_metadata_signature_covers_work_aliases_and_native_options(self):
    _, _, m, d = test_data.fixture(xml=_SLEEPING_BOXES)
    work = mjw.make_step_work(m, d)
    signature = mjw.metadata_signature((m, d, work))
    self.assertEqual(signature, mjw.metadata_signature((m, d, work)))
    self.assertFalse(any("compact_m_full" in name or "compact_d_full" in name for name, _, _ in mjw.array_fields(work)))
    m.opt.ccd_iterations += 1
    self.assertNotEqual(signature, mjw.metadata_signature((m, d, work)))
    m.opt.ccd_iterations -= 1
    work.position.efc_nnz = work.position.moment_nnz
    self.assertNotEqual(signature, mjw.metadata_signature((m, d, work)))

  @parameterized.parameters(mjw.IntegratorType.EULER, mjw.IntegratorType.IMPLICITFAST, mjw.IntegratorType.IMPLICIT)
  def test_integration_reuses_native_operands(self, integrator):
    mjm, mjd, _, _ = test_data.fixture(
      "pendula.xml", overrides={"opt.integrator": integrator, "opt.disableflags": mjw.DisableBit.CONTACT}
    )
    mjd.qvel[:] = np.linspace(-0.01, 0.01, mjm.nv)
    mujoco.mj_forward(mjm, mjd)
    m = mjw.put_model(mjm)
    eager, borrowed = (mjw.put_data(mjm, mjd) for _ in range(2))
    work = mjw.make_step_work(m, borrowed)
    for _ in range(2):
      for _, array, _ in mjw.array_fields(work):
        scalar = getattr(array.dtype, "_wp_scalar_type_", array.dtype)
        poison = float("nan") if scalar == wp.float32 else -7
        array.fill_(array.dtype(poison) if scalar != array.dtype else poison)
      mjw.step(m, eager)
      mjw.step(m, borrowed, work=work)
      for name in ("qpos", "qvel", "qacc", "qfrc_constraint"):
        actual, expected = getattr(borrowed, name).numpy(), getattr(eager, name).numpy()
        self.assertTrue(np.isfinite(actual).all() and np.isfinite(expected).all(), name)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5, equal_nan=False)

  def test_poisoned_sleeping_collision_work_matches_ordinary(self):
    mjm, mjd, m, _ = test_data.fixture(xml=_SLEEPING_BOXES)
    m.opt.broadphase = mjw.BroadphaseType.NXN
    m.opt.graph_conditional = True
    eager, borrowed = (mjw.put_data(mjm, mjd, nworld=2) for _ in range(2))
    work = mjw.make_step_work(m, borrowed)
    mjw.validate_step_work(m, borrowed, work)
    before = mjw.metadata_signature((m, borrowed, work))
    for _ in range(2):
      for _, array, _ in mjw.array_fields(work):
        scalar = getattr(array.dtype, "_wp_scalar_type_", array.dtype)
        poison = float("nan") if scalar == wp.float32 else -7
        array.fill_(array.dtype(poison) if scalar != array.dtype else poison)
      mjw.forward(m, eager)
      mjw.forward(m, borrowed, work=work)
      for name in ("qacc", "qfrc_constraint", "nefc", "nacon", "nisland"):
        actual, expected = getattr(borrowed, name).numpy(), getattr(eager, name).numpy()
        self.assertTrue(np.isfinite(actual).all() and np.isfinite(expected).all(), name)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5, equal_nan=False)
      self.assertEqual(before, mjw.metadata_signature((m, borrowed, work)))

  @parameterized.parameters(mjw.IntegratorType.EULER, mjw.IntegratorType.IMPLICIT)
  def test_sleeping_integration_does_not_reallocate_supplied_work(self, integrator):
    mjm, mjd, m, _ = test_data.fixture(
      xml="""<mujoco><option><flag sleep="enable" contact="disable"/></option>
        <worldbody><body><joint damping=".1"/><geom type="capsule" size=".1" fromto="0 0 0 0 0 1"/>
        </body></worldbody></mujoco>""",
      overrides={"opt.integrator": integrator},
    )
    eager, borrowed = (mjw.put_data(mjm, mjd) for _ in range(2))
    work = mjw.make_step_work(m, borrowed)
    work.integration.island_can_sleep.fill_(-7)
    mjw.step(m, eager)
    with mock.patch.object(wp, "ones", side_effect=AssertionError("Supplied sleep work was ignored")):
      mjw.step(m, borrowed, work=work)
    for name in ("qpos", "qvel", "tree_asleep"):
      np.testing.assert_allclose(getattr(borrowed, name).numpy(), getattr(eager, name).numpy(), rtol=1e-5, atol=1e-5)

  def test_standalone_stages_allocate_only_consumed_work(self):
    _, _, m, d = test_data.fixture("pendula.xml")
    with (
      mock.patch.object(forward, "make_step_work", side_effect=AssertionError("Stage allocated complete step work")),
      mock.patch.object(forward, "make_integration_context", side_effect=AssertionError("Stage allocated integration work")),
    ):
      with mock.patch.object(solver, "create_solver_context", side_effect=AssertionError("step1 allocated solver work")):
        mjw.step1(m, d)
      mjw.forward(m, d)
    with mock.patch.object(forward, "make_position_context", side_effect=AssertionError("step2 allocated position work")):
      mjw.step2(m, d)

  def test_native_backing_defaults_are_not_episode_reset(self):
    mjm = mujoco.MjModel.from_xml_string(_SLEEPING_BOXES)
    m, d = mjw.put_model(mjm), mjw.make_data(mjm, nworld=2)
    defaults = mjw.data_field_defaults(m, d)
    self.assertEqual(defaults["contact.efc_address"], -1)
    self.assertEqual(defaults["island_dofadr"], m.nv)
    np.testing.assert_array_equal(d.contact.efc_address.numpy(), -1)
    np.testing.assert_array_equal(d.island_dofadr.numpy(), m.nv)
    np.testing.assert_array_equal(d.qpos.numpy(), np.tile(mjm.qpos0.astype(np.float32), (2, 1)))
    d.nacon.fill_(7)
    d.ncollision.fill_(9)
    mjw.invalidate_contact_cache(d)
    np.testing.assert_array_equal(d.nacon.numpy(), [0])
    np.testing.assert_array_equal(d.ncollision.numpy(), [0])


if __name__ == "__main__":
  wp.init()
  absltest.main()
