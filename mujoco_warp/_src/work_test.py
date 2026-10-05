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
"""Ordinary numerical allocation, captured reuse and dependency boundaries."""

import ast
import dataclasses
import inspect
from pathlib import Path
from unittest import mock

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjw
from mujoco_warp import test_data
from mujoco_warp._src import collision_convex
from mujoco_warp._src import collision_core
from mujoco_warp._src import island
from mujoco_warp._src import sleep
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


class NativeAllocationTest(parameterized.TestCase):
  def test_package_has_one_numerical_path_without_managed_work(self):
    root = Path(mjw.__file__).parent
    forbidden = {
      "StepWork",
      "PositionContext",
      "IntegrationContext",
      "ConvexContext",
      "StepBindings",
      "ScratchRegistry",
      "scratch_array",
      "allocate_work_arrays",
      "make_step_work",
      "validate_step_work",
      "make_position_context",
      "make_integration_context",
      "step_work_alignments",
    }
    for path in root.rglob("*.py"):
      if path.name.endswith("_test.py") or path.name == "conftest.py" or "test_data" in path.parts:
        continue
      for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom):
          self.assertFalse((node.module or "").startswith("gpu_components"), str(path))
          self.assertFalse({item.name for item in node.names} & forbidden, str(path))
        elif isinstance(node, ast.Import):
          self.assertFalse(any(item.name.startswith("gpu_components") for item in node.names), str(path))
        elif isinstance(node, (ast.Name, ast.Attribute, ast.FunctionDef, ast.ClassDef)):
          name = node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else node.name
          self.assertNotIn(name, forbidden, str(path))
          self.assertFalse(name.endswith(("_work_layout", "_context_layout")), str(path))
    for name in ("workspace.py", "step_program.py", "step_execution.py", "scratch.py"):
      self.assertFalse((root / "_src" / name).exists(), name)
    self.assertNotIn("scratch", {field.name for field in dataclasses.fields(types.Data)})
    for path in (root.parent / "pyproject.toml", root.parent / "uv.lock"):
      self.assertNotIn("gpu-components", path.read_text())

  def test_stage_signatures_do_not_accept_borrowed_scratch(self):
    stages = (
      mjw.step,
      mjw.step1,
      mjw.step2,
      mjw.forward,
      mjw.fwd_position,
      mjw.euler,
      mjw.implicit,
      mjw.transmission,
      mjw.make_constraint,
      mjw.collision,
      solver.solve,
      solver.solve_compact,
      island.island,
      island.compute_island_mapping,
      sleep.sleep,
      mjw.deriv_smooth_vel,
      collision_core.create_collision_context,
      collision_convex.convex_narrowphase,
    )
    forbidden = {
      "work",
      "arrays",
      "convex",
      "parent",
      "efc_tree",
      "efc_nnz",
      "moment_nnz",
      "body_ncon",
      "island_can_sleep",
      "actuator_vel",
    }
    for stage in stages:
      self.assertFalse(forbidden & inspect.signature(stage).parameters.keys(), stage.__name__)
    for stage in (solver.solve, solver.solve_compact, mjw.collision):
      self.assertNotIn("ctx", inspect.signature(stage).parameters)

  def test_metadata_signature_covers_native_aliases_and_count_identity(self):
    _, _, m, d = test_data.fixture(xml=_SLEEPING_BOXES)
    signature = mjw.metadata_signature((m, d))
    m.opt.ccd_iterations += 1
    self.assertNotEqual(signature, mjw.metadata_signature((m, d)))
    m.opt.ccd_iterations -= 1
    d.cdof_dot = d.cvel
    self.assertNotEqual(signature, mjw.metadata_signature((m, d)))
    execution = dataclasses.replace(d, nworld=wp.CountParameter(d.nworld))
    signature = mjw.metadata_signature(execution)
    execution.nworld = wp.CountParameter(d.nworld)
    self.assertNotEqual(signature, mjw.metadata_signature(execution))

  @parameterized.parameters(mjw.IntegratorType.EULER, mjw.IntegratorType.IMPLICITFAST, mjw.IntegratorType.IMPLICIT)
  def test_local_temporaries_are_initialized_on_each_step(self, integrator):
    mjm, mjd, _, _ = test_data.fixture(
      "pendula.xml", overrides={"opt.integrator": integrator, "opt.disableflags": mjw.DisableBit.CONTACT}
    )
    mjd.qvel[:] = np.linspace(-0.01, 0.01, mjm.nv)
    mujoco.mj_forward(mjm, mjd)
    m = mjw.put_model(mjm)
    eager, dirty = (mjw.put_data(mjm, mjd) for _ in range(2))
    allocate = wp.empty

    def poisoned_empty(*args, **kwargs):
      array = allocate(*args, **kwargs)
      scalar = getattr(array.dtype, "_wp_scalar_type_", array.dtype)
      value = float("nan") if scalar == wp.float32 else -7
      array.fill_(array.dtype(value) if scalar != array.dtype else value)
      return array

    for _ in range(2):
      mjw.step(m, eager)
      with mock.patch.object(wp, "empty", side_effect=poisoned_empty):
        mjw.step(m, dirty)
      for name in ("qpos", "qvel", "qacc", "qfrc_constraint"):
        actual, expected = getattr(dirty, name).numpy(), getattr(eager, name).numpy()
        self.assertTrue(np.isfinite(actual).all() and np.isfinite(expected).all(), name)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5, equal_nan=False)

  def test_captured_step_reinitializes_poisoned_temporaries(self):
    if not wp.get_device().is_cuda:
      self.skipTest("Transparent allocation capture requires CUDA")
    mjm, mjd, m, _ = test_data.fixture(xml=_SLEEPING_BOXES)
    m.opt.broadphase = mjw.BroadphaseType.NXN
    m.opt.graph_conditional = True
    eager, captured = (mjw.put_data(mjm, mjd, nworld=2) for _ in range(2))
    # Compile before recording; Data and all stage temporaries keep ordinary numerical semantics.
    mjw.step(m, eager)
    captured = mjw.put_data(mjm, mjd, nworld=2)
    eager = mjw.put_data(mjm, mjd, nworld=2)
    record = lambda: wp.capture_transient(lambda: mjw.step(m, captured), assume_nonescaping=True)
    with wp.ScopedCapture(record_launches=True, record_allocations=True) as discovery:
      record()
    bindings = tuple((r, wp.empty(r.shape, dtype=r.dtype)) for r in wp.capture_get_allocations(discovery.graph))
    with wp.ScopedCapture(record_launches=True, allocation_bindings=bindings) as replay:
      record()
    for _ in range(2):
      for _, array in bindings:
        scalar = getattr(array.dtype, "_wp_scalar_type_", array.dtype)
        value = float("nan") if scalar == wp.float32 else -7
        array.fill_(array.dtype(value) if scalar != array.dtype else value)
      mjw.step(m, eager)
      wp.capture_launch(replay.graph)
      for name in ("qpos", "qvel", "qacc", "qfrc_constraint", "nefc", "nacon", "nisland"):
        actual, expected = getattr(captured, name).numpy(), getattr(eager, name).numpy()
        self.assertTrue(np.isfinite(actual).all() and np.isfinite(expected).all(), name)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5, equal_nan=False)

  def test_standalone_position_does_not_allocate_solver_state(self):
    _, _, m, d = test_data.fixture("pendula.xml")
    with mock.patch.object(solver, "create_solver_context", side_effect=AssertionError("step1 allocated solver state")):
      mjw.step1(m, d)


if __name__ == "__main__":
  absltest.main()
