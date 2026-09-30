"""CPU preparation gates for immutable metadata and unsupported native features."""

import dataclasses
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import warp as wp

from mujoco_warp import test_data
from mujoco_warp._src import solver
from mujoco_warp._src import types
from mujoco_warp._src.workspace import StepWorkspace
from mujoco_warp._src.workspace import WorkspaceField
from mujoco_warp._src.workspace import make_step_workspace
from mujoco_warp._src.workspace import step_workspace_layout


@dataclasses.dataclass
class Metadata:
  nworld: int = 17
  njmax: int = 600
  naccdmax: int = 2176
  nv: int = 12
  option: int = 0
  pair_counts: tuple = (1, 2)
  callback: object = None


class WorkspaceTest(unittest.TestCase):
  def test_cpu_template_plans_capacity_without_allocating_or_mutating(self):
    instances = []
    for count in (1, 3):
      with wp.ScopedDevice("cpu"):
        _, _, model, data = test_data.fixture(
          xml="""<mujoco><option integrator="implicitfast" solver="Newton" cone="pyramidal">
            <flag sleep="enable" island="enable"/></option><worldbody>
            <body><joint type="slide"/><geom type="sphere" size=".1"/></body>
          </worldbody></mujoco>""",
          nworld=count,
          nconmax=16,
          njmax=16,
        )
      model.opt.broadphase = types.BroadphaseType.NXN
      model.opt.disableflags |= types.DisableBit.MULTICCD
      model.opt.graph_conditional = True
      instances.append((model, data))
    model, template = instances[0]
    real_model, real_data = instances[1]
    before = StepWorkspace._layout(model), StepWorkspace._layout(template)
    with ExitStack() as stack:
      for name in (
        "empty",
        "empty_like",
        "zeros",
        "ones",
        "full",
        "clone",
        "array",
        "copy",
        "launch",
        "get_stream",
        "get_device",
      ):
        stack.enter_context(patch.object(wp, name, side_effect=AssertionError("Planning must only read metadata")))
      planned = step_workspace_layout(
        model, template, world_capacity=3, contact_capacity=real_data.naconmax, ccd_capacity=real_data.naccdmax
      )
      self.assertEqual(planned, step_workspace_layout(real_model, real_data))
      large = step_workspace_layout(model, template, world_capacity=4096, contact_capacity=8192, ccd_capacity=2048)
      self.assertEqual({spec.shape[0] for spec in large if spec.domain == "world" and spec.shape[0]}, {4096})
      self.assertEqual({spec.shape[0] for spec in large if spec.domain == "candidate"}, {8192})
      self.assertEqual({spec.shape[0] for spec in large if spec.domain == "ccd"}, {2048})
      for name, value in (
        ("world_capacity", 0),
        ("world_capacity", 2**31),
        ("world_capacity", True),
        ("contact_capacity", -1),
        ("ccd_capacity", 1.5),
      ):
        with self.assertRaises(ValueError):
          step_workspace_layout(model, template, **{name: value})
    self.assertEqual(before, (StepWorkspace._layout(model), StepWorkspace._layout(template)))

  def test_caller_scratch_binding_owns_no_allocation(self):
    specs = (
      WorkspaceField("collision_pair", (2,), wp.vec2i, "candidate"),
      WorkspaceField("collision_pairid", (2,), wp.vec2i, "candidate"),
      WorkspaceField("collision_worldid", (2,), wp.int32, "candidate"),
    )
    arrays = {spec.name: wp.empty(spec.shape, dtype=spec.dtype, device="cpu") for spec in specs}
    data = SimpleNamespace(qpos=arrays["collision_worldid"])
    with (
      patch.object(solver, "_solver_context_layout", return_value=()),
      patch.object(types, "SolverContext", return_value=SimpleNamespace()),
      patch.object(wp, "empty", side_effect=AssertionError("Unexpected allocation")),
    ):
      workspace = StepWorkspace(Metadata(), data, None, specs, None, None, arrays=arrays)
    self.assertIsNone(workspace.storage)
    self.assertEqual(workspace.memory_report()["allocation_owner"], "caller")
    for name, array in arrays.items():
      self.assertIs(workspace.arrays[name], array)
    arrays["collision_worldid"].shape = (1,)
    with self.assertRaisesRegex(ValueError, "scratch descriptors"):
      workspace.validate(workspace.model, data)

  def test_caller_scratch_invalid_descriptors_reject_before_allocating(self):
    owner = wp.empty(128, dtype=wp.uint8, device="cpu")
    spec = WorkspaceField("scratch", (2, 3), wp.float32, "world")
    data = SimpleNamespace(qpos=owner)
    valid = wp.array(ptr=owner.ptr, shape=(2, 3), strides=(32, 4), dtype=wp.float32, device="cpu")
    cases = (
      {},
      {"scratch": valid, "extra": valid},
      {"scratch": wp.empty((2, 3), dtype=wp.int32, device="cpu")},
      {"scratch": wp.empty((2, 2), dtype=wp.float32, device="cpu")},
      {"scratch": wp.array(ptr=owner.ptr, shape=(2, 3), strides=(32, 8), dtype=wp.float32, device="cpu")},
      {"scratch": wp.array(ptr=owner.ptr, shape=(2, 3), strides=(4, 4), dtype=wp.float32, device="cpu")},
    )
    with patch.object(wp, "empty", side_effect=AssertionError("Unexpected allocation")):
      for arrays in cases:
        with self.assertRaises(ValueError):
          StepWorkspace(Metadata(), data, None, (spec,), None, None, arrays=arrays)

  def test_prepared_scalar_mutation_rejected_before_capture(self):
    for owner, field in (("data", "nworld"), ("data", "njmax"), ("data", "naccdmax"), ("model", "nv"), ("model", "option")):
      with self.subTest(owner=owner, field=field):
        workspace = StepWorkspace.__new__(StepWorkspace)
        workspace.model, workspace.data = Metadata(), Metadata()
        workspace._data_layout = workspace._layout(workspace.data)
        workspace._model_layout = workspace._layout(workspace.model)
        value = getattr(workspace, owner)
        setattr(value, field, getattr(value, field) + 1)
        with self.assertRaisesRegex(ValueError, "scalar metadata"):
          workspace.validate(workspace.model, workspace.data)

  def test_foreign_model_or_data_rejected(self):
    workspace = StepWorkspace.__new__(StepWorkspace)
    workspace.model, workspace.data = Metadata(), Metadata()
    for model, data in ((Metadata(), workspace.data), (workspace.model, Metadata())):
      with self.assertRaisesRegex(ValueError, "original model"):
        workspace.validate(model, data)

  def test_prepared_topology_tuple_or_callback_change_rejected(self):
    for field, replacement in (("pair_counts", (2, 2)), ("callback", lambda *_: None)):
      workspace = StepWorkspace.__new__(StepWorkspace)
      workspace.model, workspace.data = Metadata(), Metadata()
      workspace._data_layout = workspace._layout(workspace.data)
      workspace._model_layout = workspace._layout(workspace.model)
      setattr(workspace.model, field, replacement)
      with self.assertRaisesRegex(ValueError, "scalar metadata"):
        workspace.validate(workspace.model, workspace.data)

  def test_unsupported_features_rejected_before_allocating(self):
    opt = SimpleNamespace(
      solver=types.SolverType.NEWTON,
      integrator=types.IntegratorType.IMPLICITFAST,
      cone=types.ConeType.PYRAMIDAL,
      broadphase=types.BroadphaseType.NXN,
      enableflags=types.EnableBit.SLEEP,
      disableflags=types.DisableBit.MULTICCD,
      run_collision_detection=True,
      graph_conditional=True,
    )
    model = SimpleNamespace(
      opt=opt,
      callback=types.Callback(),
      has_sdf_geom=False,
      has_fluid=False,
      nflex=0,
      ntendon=0,
      nsensor=0,
      neq=0,
      nacttrnbody=0,
      nhfield=0,
      na=0,
      nhistory=0,
    )
    data = SimpleNamespace(qpos=SimpleNamespace(device=SimpleNamespace(is_cuda=True)))
    cases = [(model, name, 1) for name in ("nflex", "ntendon", "nsensor", "neq", "nacttrnbody", "nhfield", "na", "nhistory")]
    cases += [
      (model, "has_sdf_geom", True),
      (model, "has_fluid", True),
      (opt, "solver", types.SolverType.CG),
      (opt, "integrator", types.IntegratorType.RK4),
      (opt, "cone", types.ConeType.ELLIPTIC),
      (opt, "graph_conditional", False),
      (opt, "disableflags", 0),
      (model.callback, "control", lambda *_: None),
    ]
    with (
      patch.object(wp, "get_stream", return_value=SimpleNamespace(is_capturing=False)),
      patch.object(wp, "empty", side_effect=AssertionError("Unexpected allocation")),
    ):
      for value, field, replacement in cases:
        with self.subTest(field=field):
          original = getattr(value, field)
          setattr(value, field, replacement)
          with self.assertRaises(NotImplementedError):
            make_step_workspace(model, data)
          setattr(value, field, original)

  def test_preparation_during_capture_is_rejected(self):
    data = SimpleNamespace(qpos=SimpleNamespace(device=SimpleNamespace(is_cuda=True)))
    with patch.object(wp, "get_stream", return_value=SimpleNamespace(is_capturing=True)):
      with self.assertRaisesRegex(RuntimeError, "before graph capture"):
        make_step_workspace(None, data)


if __name__ == "__main__":
  unittest.main()
