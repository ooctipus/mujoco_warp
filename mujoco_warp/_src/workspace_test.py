"""CPU preparation gates for immutable metadata and unsupported native features."""

import ast
import dataclasses
import inspect
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np
import warp as wp

from mujoco_warp import test_data
from mujoco_warp._src import constraint
from mujoco_warp._src import derivative
from mujoco_warp._src import forward
from mujoco_warp._src import island
from mujoco_warp._src import passive
from mujoco_warp._src import smooth
from mujoco_warp._src import solver
from mujoco_warp._src import support
from mujoco_warp._src import types
from mujoco_warp._src.workspace import WorkspaceFieldSpec
from mujoco_warp._src.workspace import _StepWorkspace
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
  def test_prepared_api_names_express_specs_and_execution_contract(self):
    """Keep allocation requirements and recording semantics explicit without legacy aliases."""
    import mujoco_warp as mjw

    self.assertIs(mjw.WorkspaceFieldSpec, WorkspaceFieldSpec)
    self.assertFalse(hasattr(mjw, "WorkspaceField"))
    self.assertFalse(hasattr(mjw, "StepWorkspace"))
    self.assertNotIn("StepWorkspace", vars(inspect.getmodule(_StepWorkspace)))
    self.assertEqual(
      [field.name for field in dataclasses.fields(WorkspaceFieldSpec)], ["name", "shape", "dtype", "capacity_domain"]
    )
    self.assertEqual(
      set(inspect.signature(make_step_workspace).parameters), {"model", "data", "world_live_count", "arrays", "recorder"}
    )
    self.assertFalse(hasattr(_StepWorkspace, "observe_launch"))
    self.assertIn("parameter_domains", inspect.signature(_StepWorkspace.bind_launch).parameters)

  def test_disabled_derivative_copy_preserves_inactive_rows(self):
    """Route derivative fallback through the same bounded native row-copy owner."""
    model = SimpleNamespace(
      M_fullm_i=None,
      M_fullm_j=None,
      opt=SimpleNamespace(disableflags=types.DisableBit.ACTUATION | types.DisableBit.DAMPER),
      has_fluid=False,
    )
    data = SimpleNamespace(M=wp.array(np.arange(12, dtype=np.float32).reshape(4, 3), device="cpu"))
    for prepared in (False, True):
      for live in (0, 2, 4):
        with self.subTest(prepared=prepared, live=live):
          out = wp.full((4, 3), -7.0, device="cpu")
          calls = []

          def copy(destination, source, domain):
            calls.append((destination, source, domain))
            wp.copy(destination[:live], source[:live])

          workspace = SimpleNamespace(copy=copy, validate=lambda *args: None) if prepared else None
          with patch.object(wp, "launch", side_effect=AssertionError("Unexpected derivative kernel")):
            derivative.deriv_smooth_vel(model, data, out, workspace=workspace)
          expected = np.full((4, 3), -7.0, np.float32)
          count = live if prepared else 4
          expected[:count] = data.M.numpy()[:count]
          np.testing.assert_array_equal(out.numpy(), expected)
          self.assertEqual(len(calls), int(prepared))
          if prepared:
            self.assertIs(calls[0][0], out)
            self.assertIs(calls[0][1], data.M)
            self.assertEqual(calls[0][2], "world")

  def test_disabled_features_clear_only_live_rows_with_prepared_workspace(self):
    cases = (
      (forward.fwd_actuation, 0, 0, ("act_dot", "qfrc_actuator", "actuator_force")),
      (forward.fwd_actuation, 3, types.DisableBit.ACTUATION, ("act_dot", "qfrc_actuator", "actuator_force")),
      (
        passive.passive,
        0,
        types.DisableBit.SPRING | types.DisableBit.DAMPER,
        ("qfrc_spring", "qfrc_damper", "qfrc_gravcomp", "qfrc_fluid", "qfrc_passive"),
      ),
      (smooth._rne_cacc_world, 0, types.DisableBit.GRAVITY, ("cacc",)),
    )
    for operation, nactuator, disableflags, names in cases:
      for live in (0, 2, 4):
        for prepared in (False, True):
          with self.subTest(operation=operation.__name__, live=live, prepared=prepared, nactuator=nactuator):
            arrays = {name: wp.full((4, 3), 7.0, device="cpu") for name in names}
            data = SimpleNamespace(**arrays)
            model = SimpleNamespace(nactuator=nactuator, opt=SimpleNamespace(disableflags=disableflags))
            touched = []

            def fill(array, value, domain):
              self.assertEqual(domain, "world")
              touched.append(array)
              array[:live].fill_(value)

            workspace = SimpleNamespace(fill=fill, validate=lambda *args: None) if prepared else None
            with patch.object(wp, "launch", side_effect=AssertionError("Unexpected stage work")):
              operation(model, data, workspace=workspace)
            expected = np.full((4, 3), 7, np.float32)
            expected[: live if prepared else 4] = 0
            for array in arrays.values():
              np.testing.assert_array_equal(array.numpy(), expected)
            if prepared:
              self.assertEqual([id(array) for array in touched], [id(array) for array in arrays.values()])

  def test_kinematics_declares_every_emitted_pose_launch(self):
    # Sites are allowed in prepared models, even though keyboard fixtures have none.
    model = Mock(nbranch=2, nbody=3, ngeom=4, nsite=2)
    data = Mock(nworld=7)
    emitted, observed = [], []
    workspace = SimpleNamespace(bind_launch=lambda *args: observed.append(args), validate=lambda *args: None)
    with patch.object(wp, "launch", side_effect=lambda kernel, dim, **_: emitted.append((kernel, dim, "world"))):
      smooth.kinematics(model, data, workspace=workspace)
    self.assertEqual(observed, emitted)
    self.assertEqual(observed[-1], (smooth._site_local_to_global, (7, 2), "world"))

  def test_solver_conditional_forwards_prepared_owner_to_iteration(self):
    tree = ast.parse(inspect.getsource(solver._solve))
    calls = [
      node
      for node in ast.walk(tree)
      if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "capture_while"
    ]
    self.assertEqual(len(calls), 1)
    arguments = {keyword.arg: ast.unparse(keyword.value) for keyword in calls[0].keywords}
    self.assertEqual(arguments["while_body"], "_solver_iteration")
    self.assertEqual(arguments.get("workspace"), "workspace", "Unbound WHILE nodes can over-decrement live nsolving")

  def test_launch_recorder_receives_declared_domains_and_skips_empty_launches(self):
    calls = []
    workspace = _StepWorkspace.__new__(_StepWorkspace)
    workspace.world_live_count, workspace.recorder = None, None
    workspace._execution_binding = None, None, None
    workspace.recorder = SimpleNamespace(bind_launch=lambda *args, **kwargs: calls.append((args, kwargs)))
    kernel = object()
    workspace.bind_launch(kernel, (31, 4), "world")
    workspace.bind_launch(kernel, (31, 4), "candidate")
    workspace.bind_launch(kernel, 256, None, extent_axis=None, parameter_domains={"naccdmax_in": "ccd"})
    workspace.bind_launch(kernel, (31, 0), "world")
    self.assertEqual([args[2] for args, _ in calls], ["world", "candidate", None])
    self.assertEqual(calls[2], ((kernel, 256, None), {"extent_axis": None, "parameter_domains": {"naccdmax_in": "ccd"}}))

  def test_explicit_memory_operations_preserve_eager_behavior_without_recorder(self):
    workspace = _StepWorkspace.__new__(_StepWorkspace)
    workspace.world_live_count, workspace.recorder = None, None
    workspace._execution_binding = None, None, None
    workspace.recorder = None
    source = wp.array(np.arange(12, dtype=np.float32).reshape(3, 4), device="cpu")
    destination = wp.zeros((3, 4), dtype=wp.float32, device="cpu")
    workspace.copy(destination, source, "world")
    np.testing.assert_array_equal(destination.numpy(), source.numpy())
    self.assertIs(workspace.fill(destination, 7, "world"), destination)
    np.testing.assert_array_equal(destination.numpy(), np.full((3, 4), 7, np.float32))

  def test_recorder_memory_failure_propagates_without_dense_fallback(self):
    workspace = _StepWorkspace.__new__(_StepWorkspace)
    workspace.world_live_count, workspace.recorder = None, None
    workspace._execution_binding = None, None, None
    array = SimpleNamespace(size=12, fill_=lambda *_: self.fail("Unexpected dense fill"))

    def fail(*args):
      self.assertEqual(args[-1], "world")
      raise ArithmeticError("Bounded operation failed")

    workspace.recorder = SimpleNamespace(fill=fail, copy=fail)
    with patch.object(wp, "copy", side_effect=AssertionError("Unexpected dense copy")):
      for operation in (lambda: workspace.fill(array, 0, "world"), lambda: workspace.copy(array, array, "world")):
        with self.assertRaisesRegex(ArithmeticError, "Bounded operation failed"):
          operation()
      empty = SimpleNamespace(size=0)
      workspace.fill(empty, 0, "world")
      workspace.copy(empty, empty, "world")

  def test_incomplete_recorder_rejected_before_scratch_planning(self):
    data = SimpleNamespace(qpos=SimpleNamespace(device=SimpleNamespace(is_cuda=True)))
    with (
      patch.object(wp, "get_stream", return_value=SimpleNamespace(is_capturing=False)),
      patch("mujoco_warp._src.workspace.step_workspace_layout", side_effect=AssertionError("Unexpected scratch planning")),
    ):
      with self.assertRaisesRegex(TypeError, "bind_launch, fill and copy"):
        make_step_workspace(None, data, recorder=SimpleNamespace(bind_launch=lambda *_: None))

  def test_dynamic_execution_requires_count_and_recorder_together(self):
    """Reject incomplete execution bindings before planning or allocating scratch."""
    data = SimpleNamespace(qpos=SimpleNamespace(device=SimpleNamespace(is_cuda=True)))
    recorder = SimpleNamespace(bind_launch=lambda *_: None, fill=lambda *_: None, copy=lambda *_: None)
    with (
      patch.object(wp, "get_stream", return_value=SimpleNamespace(is_capturing=False)),
      patch("mujoco_warp._src.workspace.step_workspace_layout", side_effect=AssertionError("Unexpected scratch planning")),
    ):
      for arguments in ({"world_live_count": object()}, {"recorder": recorder}):
        with self.subTest(arguments=arguments), self.assertRaisesRegex(ValueError, "together"):
          make_step_workspace(None, data, **arguments)

  def test_recording_rejects_detached_dynamic_execution_binding(self):
    """Reject a detached count recorder before a partial-capacity program is recorded."""
    workspace = _StepWorkspace.__new__(_StepWorkspace)
    workspace.world_live_count, workspace.recorder = None, None
    workspace._execution_binding = None, None, None
    workspace.model, workspace.data = Metadata(), Metadata()
    workspace.world_live_count, workspace.recorder = object(), None
    workspace._model_layout, workspace._data_layout = workspace._layout(workspace.model), workspace._layout(workspace.data)
    workspace.arrays, workspace._scratch_layout = {}, ()
    workspace.device = SimpleNamespace(captures={})
    with patch.object(wp, "get_stream", return_value=None), self.assertRaisesRegex(ValueError, "together"):
      workspace.validate(workspace.model, workspace.data)

  def test_every_exported_prepared_entry_validates_or_rejects_before_work(self):
    """Keep the public prepared-stage surface complete and guard every callable before writes."""
    import mujoco_warp as mjw

    expected = {
      "step",
      "collision",
      "nxn_broadphase",
      "primitive_narrowphase",
      "make_constraint",
      "deriv_smooth_vel",
      "euler",
      "forward",
      "fwd_acceleration",
      "fwd_actuation",
      "fwd_kinematics",
      "fwd_position",
      "fwd_velocity",
      "implicit",
      "rungekutta4",
      "island",
      "passive",
      "energy_vel",
      "sensor_acc",
      "sensor_pos",
      "com_pos",
      "com_vel",
      "crb",
      "kinematics",
      "rne",
      "rne_postconstraint",
      "transmission",
      "solve",
      "mul_m",
      "xfrc_accumulate",
    }
    exported = {
      name: function
      for name, function in vars(mjw).items()
      if inspect.isfunction(function) and "workspace" in inspect.signature(function).parameters
    }
    self.assertEqual(set(exported), expected)
    unsupported = {"euler", "rungekutta4", "rne_postconstraint", "energy_vel"}
    for name, program in exported.items():
      validate = Mock(side_effect=ValueError("invalid prepared binding"))
      arguments = [
        object()
        for parameter in inspect.signature(program).parameters.values()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
      ]
      with self.subTest(program=name), ExitStack() as stack:
        for operation in ("launch", "launch_tiled", "empty", "empty_like", "zeros", "clone", "copy"):
          stack.enter_context(patch.object(wp, operation, side_effect=AssertionError("stage work before validation")))
        if name in unsupported:
          with self.assertRaisesRegex(NotImplementedError, "Prepared"):
            program(*arguments, workspace=SimpleNamespace(validate=validate))
          validate.assert_not_called()
        else:
          with self.assertRaisesRegex(ValueError, "invalid prepared binding"):
            program(*arguments, workspace=SimpleNamespace(validate=validate))
          validate.assert_called_once()

  def test_prepared_position_rejects_unprepared_factorization(self):
    """Reject the eager default factorization path before any position work is emitted."""
    workspace = SimpleNamespace(validate=lambda *args: None)
    with patch.object(wp, "launch", side_effect=AssertionError("work before rejection")):
      with self.assertRaisesRegex(NotImplementedError, "factorize=False"):
        forward.fwd_position(object(), object(), workspace=workspace)

  def test_standalone_stages_reuse_prepared_scratch_when_optional_arguments_omitted(self):
    """Use the workspace's actuator and island scratch without hidden stage allocations."""
    parent, velocity = object(), object()
    workspace = SimpleNamespace(
      arrays={"island_parent": parent, "actuator_vel": velocity},
      validate=lambda *args: None,
      bind_launch=lambda *args: None,
      fill=lambda *args: None,
    )
    with (
      patch.object(island, "direct_dsu") as launch,
      patch.object(wp, "empty", side_effect=AssertionError("scratch allocated")),
    ):
      island.island(SimpleNamespace(ntree=2), SimpleNamespace(nworld=3), workspace=workspace)
      self.assertIs(launch.call_args.args[2], parent)
    model = Mock(opt=SimpleNamespace(disableflags=0, timestep=None), nactuator=2, has_fluid=False)
    data = Mock(nworld=3)
    with patch.object(wp, "launch") as launch, patch.object(wp, "empty", side_effect=AssertionError("scratch allocated")):
      derivative.deriv_smooth_vel(model, data, object(), workspace=workspace)
      self.assertIs(launch.call_args_list[0].kwargs["outputs"][0], velocity)
    for function, name, position in (
      (constraint.make_constraint, "efc_nnz", -1),
      (smooth.transmission, "moment_nnz", -1),
    ):
      workspace.arrays[name] = object()
      workspace.fill = Mock()
      model = Mock(opt=SimpleNamespace(solver=types.SolverType.NEWTON, disableflags=types.DisableBit.CONSTRAINT))
      with (
        patch.object(wp, "launch", side_effect=RuntimeError("first stage launch")) as launch,
        patch.object(wp, "empty", side_effect=AssertionError("scratch allocated")),
        patch.object(wp, "zeros", side_effect=AssertionError("scratch allocated")),
        self.assertRaisesRegex(RuntimeError, "first stage launch"),
      ):
        function(model, data, workspace=workspace)
      self.assertIs(launch.call_args.kwargs["inputs"][position], workspace.arrays[name])
      if name == "moment_nnz":
        workspace.fill.assert_called_once_with(workspace.arrays[name], 0, "world")

  def test_prepared_stages_reject_alternate_scratch_before_writing_or_launching(self):
    """One prepared scratch owner cannot be bypassed through an optional eager argument."""
    for function, arguments, parameter, field in (
      (constraint.make_constraint, (object(), object()), "efc_nnz", "efc_nnz"),
      (smooth.transmission, (object(), object()), "moment_nnz", "moment_nnz"),
      (island.island, (object(), object()), "parent", "island_parent"),
      (derivative.deriv_smooth_vel, (object(), object(), object()), "actuator_vel", "actuator_vel"),
    ):
      workspace = SimpleNamespace(validate=lambda *args: None, arrays={field: object()}, fill=Mock())
      with (
        self.subTest(function=function.__name__),
        patch.object(wp, "launch", side_effect=AssertionError("unadmitted scratch touched")),
        self.assertRaisesRegex(ValueError, "workspace .* scratch"),
      ):
        function(*arguments, **{parameter: object()}, workspace=workspace)
      workspace.fill.assert_not_called()

  def test_prepared_inertia_multiply_rejects_dense_override_and_does_not_allocate_unused_skip(self):
    """Reject dense overrides before access and bind admitted scalar inertia launches."""
    workspace = SimpleNamespace(validate=Mock(), bind_launch=Mock())
    model, data = Mock(nv=2), Mock(nworld=7, M=SimpleNamespace(ndim=2))
    with (
      patch.object(wp, "launch", side_effect=AssertionError("unbounded dense access")) as launch,
      patch.object(wp, "empty", side_effect=AssertionError("hidden allocation")),
      self.assertRaisesRegex(NotImplementedError, "dense block inertia"),
    ):
      support.mul_m(model, data, object(), object(), M=SimpleNamespace(ndim=3), workspace=workspace)
    launch.assert_not_called()
    workspace.bind_launch.assert_not_called()
    with patch.object(wp, "launch") as launch, patch.object(wp, "empty", side_effect=AssertionError("hidden allocation")):
      support.mul_m(model, data, object(), object(), workspace=workspace)
    self.assertIsNone(launch.call_args.kwargs["inputs"][-1])
    workspace.bind_launch.assert_called_once_with(launch.call_args.args[0], (7, 2), "world")

  def test_prepared_execution_binding_cannot_change_mode_count_or_recorder(self):
    """Reject dynamic-to-fixed escape and rebinding while permitting count value updates."""
    count = wp.zeros(1, dtype=wp.int32, device="cpu")
    recorder = SimpleNamespace(bind_launch=lambda *a, **kw: None, fill=lambda *a: None, copy=lambda *a: None)
    specs = (
      WorkspaceFieldSpec("collision_pair", (2,), wp.vec2i, "candidate"),
      WorkspaceFieldSpec("collision_pairid", (2,), wp.vec2i, "candidate"),
      WorkspaceFieldSpec("collision_worldid", (2,), wp.int32, "candidate"),
    )
    arrays = {spec.name: wp.empty(spec.shape, dtype=spec.dtype, device="cpu") for spec in specs}
    data = SimpleNamespace(qpos=count)
    with (
      patch.object(solver, "_solver_context_layout", return_value=()),
      patch.object(types, "SolverContext", return_value=SimpleNamespace()),
    ):
      dynamic = _StepWorkspace(Metadata(), data, count, specs, None, None, arrays=arrays, recorder=recorder)
      fixed = _StepWorkspace(Metadata(), data, None, specs, None, None, arrays=arrays)
    with patch.object(wp, "get_stream", return_value=None):
      for next_count, next_recorder in (
        (None, None),
        (wp.zeros(1, dtype=wp.int32, device="cpu"), recorder),
        (count, SimpleNamespace(**vars(recorder))),
      ):
        dynamic.world_live_count, dynamic.recorder = next_count, next_recorder
        with self.assertRaisesRegex(ValueError, "execution binding changed"):
          dynamic.validate(dynamic.model, data)
      dynamic.world_live_count, dynamic.recorder = count, recorder
      count.fill_(1)
      dynamic.validate(dynamic.model, data)
      for name, changed in (("shape", (0,)), ("strides", (8,)), ("dtype", wp.int64)):
        original = getattr(count, name)
        try:
          setattr(count, name, changed)
          with self.assertRaisesRegex(ValueError, "execution binding changed"):
            dynamic.validate(dynamic.model, data)
        finally:
          setattr(count, name, original)
      dynamic.recorder = None
      with self.assertRaisesRegex(ValueError, "together"):
        dynamic.validate(dynamic.model, data)
      dynamic.recorder = recorder
      dynamic.validate(dynamic.model, data)
      fixed.world_live_count, fixed.recorder = count, recorder
      with self.assertRaisesRegex(ValueError, "execution binding changed"):
        fixed.validate(fixed.model, data)

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
    before = _StepWorkspace._layout(model), _StepWorkspace._layout(template)
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
      self.assertEqual({spec.shape[0] for spec in large if spec.capacity_domain == "world" and spec.shape[0]}, {4096})
      self.assertEqual({spec.shape[0] for spec in large if spec.capacity_domain == "candidate"}, {8192})
      self.assertEqual({spec.shape[0] for spec in large if spec.capacity_domain == "ccd"}, {2048})
      for name, value in (
        ("world_capacity", 0),
        ("world_capacity", 2**31),
        ("world_capacity", True),
        ("contact_capacity", -1),
        ("ccd_capacity", 1.5),
      ):
        with self.assertRaises(ValueError):
          step_workspace_layout(model, template, **{name: value})
    self.assertEqual(before, (_StepWorkspace._layout(model), _StepWorkspace._layout(template)))

  def test_caller_scratch_binding_owns_no_allocation(self):
    specs = (
      WorkspaceFieldSpec("collision_pair", (2,), wp.vec2i, "candidate"),
      WorkspaceFieldSpec("collision_pairid", (2,), wp.vec2i, "candidate"),
      WorkspaceFieldSpec("collision_worldid", (2,), wp.int32, "candidate"),
    )
    arrays = {spec.name: wp.empty(spec.shape, dtype=spec.dtype, device="cpu") for spec in specs}
    data = SimpleNamespace(qpos=arrays["collision_worldid"])
    with (
      patch.object(solver, "_solver_context_layout", return_value=()),
      patch.object(types, "SolverContext", return_value=SimpleNamespace()),
      patch.object(wp, "empty", side_effect=AssertionError("Unexpected allocation")),
    ):
      workspace = _StepWorkspace(Metadata(), data, None, specs, None, None, arrays=arrays)
    self.assertIsNone(workspace.storage)
    report = workspace.memory_report()
    self.assertEqual(report["allocation_owner"], "caller")
    for name in ("solver_model", "solver_data", "solver_context", "collision"):
      self.assertFalse(hasattr(workspace, name), "Derived stage views belong to the engine, not caller configuration")
    self.assertIsNone(report["physical_scratch_bytes"])
    self.assertEqual(report["payload_bytes"], 40)
    for field in report["fields"]:
      self.assertEqual(field["capacity_domain"], "candidate")
      self.assertIsNone(field["allocation_offset_bytes"])
      self.assertFalse({"scope", "offset", "bytes"} & field.keys())
    for name, array in arrays.items():
      self.assertIs(workspace.arrays[name], array)
    arrays["collision_worldid"].shape = (1,)
    with self.assertRaisesRegex(ValueError, "scratch descriptors"):
      workspace.validate(workspace.model, data)

  def test_caller_scratch_invalid_descriptors_reject_before_allocating(self):
    owner = wp.empty(128, dtype=wp.uint8, device="cpu")
    spec = WorkspaceFieldSpec("scratch", (2, 3), wp.float32, "world")
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
          _StepWorkspace(Metadata(), data, None, (spec,), None, None, arrays=arrays)

  def test_prepared_scalar_mutation_rejected_before_capture(self):
    for owner, field in (("data", "nworld"), ("data", "njmax"), ("data", "naccdmax"), ("model", "nv"), ("model", "option")):
      with self.subTest(owner=owner, field=field):
        workspace = _StepWorkspace.__new__(_StepWorkspace)
        workspace.world_live_count, workspace.recorder = None, None
        workspace._execution_binding = None, None, None
        workspace.model, workspace.data = Metadata(), Metadata()
        workspace._data_layout = workspace._layout(workspace.data)
        workspace._model_layout = workspace._layout(workspace.model)
        value = getattr(workspace, owner)
        setattr(value, field, getattr(value, field) + 1)
        with self.assertRaisesRegex(ValueError, "scalar metadata"):
          workspace.validate(workspace.model, workspace.data)

  def test_foreign_model_or_data_rejected(self):
    workspace = _StepWorkspace.__new__(_StepWorkspace)
    workspace.world_live_count, workspace.recorder = None, None
    workspace._execution_binding = None, None, None
    workspace.model, workspace.data = Metadata(), Metadata()
    for model, data in ((Metadata(), workspace.data), (workspace.model, Metadata())):
      with self.assertRaisesRegex(ValueError, "original model"):
        workspace.validate(model, data)

  def test_nested_list_array_rebinding_is_part_of_immutable_metadata(self):
    """Inspect supported list containers recursively instead of trusting their Python identity."""
    for owner in ("model", "data"):
      workspace = _StepWorkspace.__new__(_StepWorkspace)
      workspace.world_live_count, workspace.recorder = None, None
      workspace._execution_binding = None, None, None
      workspace.model, workspace.data = Metadata(), Metadata()
      nested = [wp.zeros(1, dtype=float, device="cpu")]
      getattr(workspace, owner).pair_counts = [nested]
      workspace._data_layout = workspace._layout(workspace.data)
      workspace._model_layout = workspace._layout(workspace.model)
      nested[0] = wp.ones(1, dtype=float, device="cpu")
      with self.subTest(owner=owner), self.assertRaisesRegex(ValueError, "scalar metadata"):
        workspace.validate(workspace.model, workspace.data)

  def test_prepared_topology_tuple_or_callback_change_rejected(self):
    for field, replacement in (("pair_counts", (2, 2)), ("callback", lambda *_: None)):
      workspace = _StepWorkspace.__new__(_StepWorkspace)
      workspace.world_live_count, workspace.recorder = None, None
      workspace._execution_binding = None, None, None
      workspace.model, workspace.data = Metadata(), Metadata()
      workspace._data_layout = workspace._layout(workspace.data)
      workspace._model_layout = workspace._layout(workspace.model)
      setattr(workspace.model, field, replacement)
      with self.assertRaisesRegex(ValueError, "scalar metadata"):
        workspace.validate(workspace.model, workspace.data)

  def test_blocked_scratch_alignment_rejects_base_and_world_stride(self):
    """Reject misaligned blocked matrices without restricting dense scalar counters."""
    specs = (
      WorkspaceFieldSpec("collision_pair", (2,), wp.vec2i, "candidate"),
      WorkspaceFieldSpec("collision_pairid", (2,), wp.vec2i, "candidate"),
      WorkspaceFieldSpec("collision_worldid", (2,), wp.int32, "candidate"),
      WorkspaceFieldSpec("solver.h", (2, 64, 64), wp.float32, "world"),
      WorkspaceFieldSpec("solver.hfactor", (2, 64, 64), wp.float32, "world"),
      WorkspaceFieldSpec("moment_nnz", (2,), wp.int32, "world"),
    )
    owner = wp.empty(65536, dtype=wp.uint8, device="cpu")
    baseline = {spec.name: wp.empty(spec.shape, dtype=spec.dtype, device="cpu") for spec in specs}
    data, model = SimpleNamespace(qpos=baseline["moment_nnz"]), Metadata(nv=64)
    for name in ("solver.h", "solver.hfactor"):
      for base, stride in ((4, 16384), (0, 16388)):
        arrays = dict(baseline)
        arrays[name] = wp.array(
          ptr=owner.ptr + base, shape=(2, 64, 64), strides=(stride, 256, 4), dtype=wp.float32, device="cpu"
        )
        with (
          self.subTest(name=name, base=base, stride=stride),
          patch.object(solver, "_solver_context_layout", return_value=()),
          patch.object(types, "SolverContext", return_value=SimpleNamespace()),
          self.assertRaisesRegex(ValueError, "16-byte"),
        ):
          _StepWorkspace(model, data, None, specs, model, None, arrays=arrays)
    with (
      patch.object(solver, "_solver_context_layout", return_value=()),
      patch.object(types, "SolverContext", return_value=SimpleNamespace()),
      patch.object(wp, "empty", side_effect=AssertionError("Unexpected allocation")),
    ):
      workspace = _StepWorkspace(model, data, None, specs, model, None, arrays=baseline)
    self.assertEqual(workspace.arrays["moment_nnz"].strides, (4,))
    # The nonblocked path does not request aligned=True; an absent hfactor is valid.
    small = (
      *specs[:3],
      WorkspaceFieldSpec("solver.h", (2, 2, 2), wp.float32, "world"),
      WorkspaceFieldSpec("solver.hfactor", (2, 0, 0), wp.float32, "world"),
      specs[-1],
    )
    arrays = {spec.name: wp.empty(spec.shape, dtype=spec.dtype, device="cpu") for spec in small}
    arrays["solver.h"] = wp.array(ptr=owner.ptr + 4, shape=(2, 2, 2), strides=(20, 8, 4), dtype=wp.float32, device="cpu")
    with (
      patch.object(solver, "_solver_context_layout", return_value=()),
      patch.object(types, "SolverContext", return_value=SimpleNamespace()),
    ):
      _StepWorkspace(Metadata(nv=2), data, None, small, Metadata(nv=2), None, arrays=arrays)

  def test_compact_data_matrix_alignment_rejects_each_address_axis(self):
    """Reject misaligned compact matrix bases/strides before scratch binding."""
    owner = wp.empty(65536, dtype=wp.uint8, device="cpu")
    valid = wp.array(ptr=owner.ptr, shape=(2, 64, 64), strides=(16384, 256, 4), dtype=wp.float32, device="cpu")
    data = SimpleNamespace(qpos=SimpleNamespace(device=SimpleNamespace(is_cuda=True)), cM=valid, cqLD=valid)
    with (
      patch.object(wp, "get_stream", return_value=SimpleNamespace(is_capturing=False)),
      patch("mujoco_warp._src.workspace.step_workspace_layout", return_value=()),
      patch.object(solver, "_compact_solver_views", return_value=(None, None)),
      patch("mujoco_warp._src.workspace._StepWorkspace", return_value="bound") as bind,
    ):
      for name in ("cM", "cqLD"):
        for base, world_stride, row_stride in ((4, 16384, 256), (0, 16388, 256), (0, 16640, 260)):
          setattr(
            data,
            name,
            wp.array(
              ptr=owner.ptr + base, shape=(2, 64, 64), strides=(world_stride, row_stride, 4), dtype=wp.float32, device="cpu"
            ),
          )
          with self.subTest(name=name, base=base, stride=world_stride):
            with self.assertRaisesRegex(ValueError, "16-byte"):
              make_step_workspace(None, data)
            bind.assert_not_called()
          bind.reset_mock()
          setattr(data, name, valid)
      self.assertEqual(make_step_workspace(None, data), "bound")
      data.cM = data.cqLD = wp.empty((2, 0, 0), dtype=wp.float32, device="cpu")
      self.assertEqual(make_step_workspace(None, data), "bound")

  def test_implicit_fast_free_body_path_rejected_before_planning(self):
    """Reject the actual unbound free-body path before scratch planning or allocation."""
    with wp.ScopedDevice("cpu"):
      _, _, model, data = test_data.fixture(
        xml="""<mujoco><option integrator="implicitfast" solver="Newton" cone="pyramidal">
          <flag sleep="enable" island="enable"/></option><worldbody>
          <body><freejoint/><geom type="sphere" size=".1"/></body>
        </worldbody></mujoco>""",
        nworld=1,
        nconmax=16,
        njmax=16,
      )
    np.testing.assert_array_equal(model.body_freeadr.numpy(), [1])
    model.opt.broadphase = types.BroadphaseType.NXN
    model.opt.disableflags |= types.DisableBit.MULTICCD
    model.opt.graph_conditional = True
    base = model.opt.disableflags
    flags = (types.DisableBit.ACTUATION, types.DisableBit.SPRING, types.DisableBit.DAMPER)
    with (
      patch.object(wp, "empty", side_effect=AssertionError("Unexpected scratch allocation")),
      patch.object(solver, "_compact_solver_views", side_effect=AssertionError("Unexpected scratch planning")),
    ):
      for mask in range(7):
        model.opt.disableflags = base | sum(int(flag) for bit, flag in enumerate(flags) if mask & (1 << bit))
        with self.subTest(disabled=mask), self.assertRaisesRegex(NotImplementedError, "free-body solves"):
          step_workspace_layout(model, data, world_capacity=31)
    # This exact mask bypasses both specialized launches in forward.implicit.
    model.opt.disableflags = base | flags[0] | flags[1] | flags[2]
    self.assertTrue(step_workspace_layout(model, data, world_capacity=31))

  def test_unbound_optional_programs_rejected_with_real_cpu_metadata(self):
    """Conservatively admit only optional branches with prepared count bindings."""
    slide = '<body><joint type="slide"/><geom type="sphere" size=".1"/></body>'
    ball = '<body><joint type="ball" limited="true" range="0 45"/><geom type="sphere" size=".1"/></body>'
    chain = '<body><joint/><geom type="sphere" size=".1"/>'
    cases = (
      ("static", '<body><geom type="sphere" size=".1"/></body>', "dense", "dynamic tree"),
      ("ball", ball, "dense", "ball-joint limits"),
      ("surface", slide, "dense", "surface velocity"),
      ("adhesion", slide, "dense", "passive adhesion"),
      ("rne", slide, "dense", "postconstraint inverse dynamics"),
      ("dense", slide * 60, "dense", "dense full Jacobians"),
      ("gathered", chain * 10 + "</body>" * 10, "sparse", "inertia factorizations"),
      ("ldl", chain * 80 + "</body>" * 80, "sparse", "inertia factorizations"),
      ("sparse", slide * 108, "sparse", None),
    )
    for feature, body, jacobian, message in cases:
      with self.subTest(feature=feature), wp.ScopedDevice("cpu"):
        _, _, model, data = test_data.fixture(
          xml=f"""<mujoco><default><geom contype="0" conaffinity="0"/></default>
            <option integrator="implicitfast" solver="Newton" cone="pyramidal" jacobian="{jacobian}">
            <flag sleep="enable" island="enable"/></option><worldbody>{body}</worldbody></mujoco>""",
          nworld=1,
          nconmax=16,
          njmax=16,
        )
        model.opt.broadphase = types.BroadphaseType.NXN
        model.opt.disableflags |= types.DisableBit.MULTICCD
        model.opt.graph_conditional = True
        if feature == "static":
          self.assertEqual(model.ntree, 0)
        elif feature == "ball":
          np.testing.assert_array_equal(model.jnt_limited_ball_adr.numpy(), [0])
        elif feature == "surface":
          model.flg_surfacevel = True
        elif feature == "adhesion":
          model.flg_adhesion = True
          self.assertEqual(model.nacttrnbody, 0)
        elif feature == "rne":
          model.opt.run_rne_postconstraint = True
        elif feature == "dense":
          self.assertFalse(model.is_sparse)
          self.assertEqual(data.nvmax_pad, 64)
        elif feature == "gathered":
          self.assertTrue(any(tile.elemid.size for tile in model.M_tiles))
        elif feature == "ldl":
          self.assertGreater(data.qLD.shape[1], model.qLD_block_total)
        if message is None:
          self.assertTrue(step_workspace_layout(model, data, world_capacity=31))
          continue
        with (
          patch.object(wp, "empty", side_effect=AssertionError("Unexpected allocation")),
          patch.object(solver, "_compact_solver_views", side_effect=AssertionError("Unexpected scratch planning")),
          self.assertRaisesRegex(NotImplementedError, message),
        ):
          step_workspace_layout(model, data, world_capacity=31)
        if feature in ("dense", "static"):
          continue  # Sparse 108-DOF positive case is an independently authored model.
        if feature == "ball":
          model.opt.disableflags |= types.DisableBit.LIMIT
        elif feature in ("surface", "adhesion"):
          model.opt.disableflags |= types.DisableBit.CONTACT
        elif feature == "rne":
          model.opt.run_rne_postconstraint = False
        else:
          model.opt.disableflags |= types.DisableBit.ACTUATION | types.DisableBit.SPRING | types.DisableBit.DAMPER
        self.assertTrue(step_workspace_layout(model, data, world_capacity=31))

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
      ncam=0,
      nlight=0,
      neq=0,
      nacttrnbody=0,
      nhfield=0,
      na=0,
      nhistory=0,
    )
    data = SimpleNamespace(qpos=SimpleNamespace(device=SimpleNamespace(is_cuda=True)))
    cases = [
      (model, name, 1)
      for name in ("nflex", "ntendon", "nsensor", "ncam", "nlight", "neq", "nacttrnbody", "nhfield", "na", "nhistory")
    ]
    cases += [
      (model, "has_sdf_geom", True),
      (model, "has_fluid", True),
      (opt, "solver", types.SolverType.CG),
      (opt, "integrator", types.IntegratorType.RK4),
      (opt, "cone", types.ConeType.ELLIPTIC),
      (opt, "enableflags", types.EnableBit.SLEEP | types.EnableBit.ENERGY),
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
