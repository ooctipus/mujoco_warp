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

"""CPU gates for one complete scratch binding and immutable captured operands."""

import ast
import dataclasses
import unittest
import weakref
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from unittest.mock import PropertyMock
from unittest.mock import patch

import mujoco
import numpy as np
import warp as wp
from gpu_components import scratch as scratch_ops
from gpu_components import scratch_data
from gpu_components.field_data import FieldStorage
from gpu_components.field_data import FieldView
from gpu_components.graph_data import GraphUpdateTable

import mujoco_warp as mjw
from mujoco_warp import test_data
from mujoco_warp._src import derivative
from mujoco_warp._src import forward
from mujoco_warp._src import io as native_io
from mujoco_warp._src import passive
from mujoco_warp._src import sleep
from mujoco_warp._src import smooth
from mujoco_warp._src import solver
from mujoco_warp._src import step_program
from mujoco_warp._src import types
from mujoco_warp._src import warp_util
from mujoco_warp._src import workspace as native_workspace


@dataclasses.dataclass
class Metadata:
  nworld: int = 4
  naconmax: int = 4
  naccdmax: int = 4
  option: int = 0
  qpos: object = None
  cM: object = None
  cqLD: object = None
  nacon: object = None
  ncollision: object = None
  scratch: object = None
  nested: object = None


def _fixture():
  """Real CPU arrays and passive storage records; only CUDA admission is replaced."""
  stores, columns = [], {}
  parameters = tuple(wp.CountParameter(4) for _ in range(3))
  registry = scratch_ops.allocate_registry("cpu")
  for i, parameter in enumerate(parameters):
    name = f"domain{i}"
    array = scratch_ops.scratch_array(registry, name, (parameter, 3), float)
    count, ready = (wp.zeros(1, dtype=int, device="cpu") for _ in range(2))
    view = FieldView(name, array, 12, None, array)
    stores.append(
      FieldStorage(
        4,
        count,
        count.device,
        None,
        0,
        {},
        {name: array},
        {name: view},
        {(array.ptr, array.dtype, array.shape, array.strides): view},
        {},
        weakref.WeakSet(),
        weakref.WeakSet(),
        ready_rows=4,
        ready_count=ready,
      )
    )
    columns[name] = array
  columns["counter"] = scratch_ops.scratch_array(registry, "counter", (1,), int)
  scratch_ops.freeze(registry)
  scratch_ops.prepare(registry, columns)
  bindings = step_program.StepBindings(*stores, fixed_arrays=(columns["counter"],))
  empty = wp.empty((4, 0, 0), dtype=float, device="cpu")
  data = Metadata(
    qpos=wp.zeros((4, 2), dtype=float, device="cpu"),
    cM=empty,
    cqLD=empty,
    nacon=wp.zeros(1, dtype=int, device="cpu"),
    ncollision=wp.zeros(1, dtype=int, device="cpu"),
  )
  return Metadata(), data, bindings, registry, parameters


@contextmanager
def _cpu_preparation(*, capturing=False):
  with (
    patch.object(type(wp.get_device("cpu")), "is_cuda", new_callable=PropertyMock, return_value=True),
    patch.object(wp, "get_stream", return_value=SimpleNamespace(is_capturing=capturing)),
    patch.object(native_workspace, "validate_step_model"),
    patch.object(
      native_io,
      "array_fields",
      side_effect=lambda data: tuple((name, value, ()) for name, value in vars(data).items() if isinstance(value, wp.array)),
    ),
  ):
    yield


def _prepare(fixture):
  model, data, bindings, registry, parameters = fixture
  with _cpu_preparation():
    return native_workspace.make_step_workspace(model, data, bindings=bindings, scratch=registry, count_parameters=parameters)


def _updates(bindings):
  count = bindings.world_storage.protected_count
  return GraphUpdateTable(count, 4, count.device, 32, 120, count, count, count, None, {})


@contextmanager
def _capture(bindings):
  bindings.updates = _updates(bindings)

  class Graph:
    pass

  graph = Graph()
  graph.device = bindings.world_storage.device
  with patch.object(step_program.graph_ops, "current_capture", return_value=graph):
    yield graph


class PreparedBindingTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    wp.init()

  def test_complete_binding_preserves_count_identity_and_borrows_arrays(self):
    fixture = _fixture()
    model, data, bindings, registry, parameters = fixture
    with patch.object(wp, "empty", side_effect=AssertionError("binding must not allocate")):
      workspace = _prepare(fixture)
    self.assertIs(workspace.data, data)
    self.assertIsNone(data.scratch)
    self.assertIs(workspace.execution_data.scratch, registry)
    self.assertIs(workspace.execution_data.qpos, data.qpos)
    for (parameter, source), expected, count in zip(
      step_program._count_sources(workspace.execution_data, bindings),
      parameters,
      (bindings.world_storage.protected_count, bindings.contact_storage.ready_count, bindings.ccd_storage.ready_count),
      strict=True,
    ):
      self.assertIs(parameter, expected)
      self.assertIs(source, count)
    with patch.object(step_program.graph_ops, "current_capture", return_value=None):
      step_program.validate_step_workspace(workspace, model, data)

  def test_incomplete_registry_and_wrong_count_domains_reject(self):
    for problem in ("discovering", "frozen", "missing", "foreign_count", "wrong_domain", "fixed_missing", "wrong_dtype"):
      fixture = _fixture()
      _, _, bindings, registry, parameters = fixture
      if problem in ("discovering", "frozen"):
        registry.state = problem
      elif problem == "missing":
        del registry.columns["domain0"]
      elif problem == "foreign_count":
        registry.declarations["domain0"].shape = (wp.CountParameter(4), 3)
      elif problem == "wrong_domain":
        registry.declarations["domain0"].shape = (parameters[1], 3)
      elif problem == "fixed_missing":
        bindings.fixed_arrays = ()
      else:
        registry.columns["counter"] = wp.zeros(1, dtype=float, device="cpu")
        bindings.fixed_arrays = (registry.columns["counter"],)
      with self.subTest(problem=problem), self.assertRaises((ValueError, RuntimeError, KeyError)):
        _prepare(fixture)

  def test_domains_need_distinct_parameters_and_matching_capacities(self):
    for problem in ("same_identity", "wrong_bound", "wrong_capacity"):
      model, data, bindings, registry, parameters = _fixture()
      if problem == "same_identity":
        parameters = (parameters[0], parameters[0], parameters[2])
      elif problem == "wrong_bound":
        parameters = (parameters[0], wp.CountParameter(5), parameters[2])
      else:
        bindings.ccd_storage.capacity = 5
      with self.subTest(problem=problem), self.assertRaises(ValueError):
        _prepare((model, data, bindings, registry, parameters))

  def test_counter_cannot_alias_an_execution_count(self):
    for declared in (False, True):
      fixture = _fixture()
      _, _, bindings, registry, _ = fixture
      count = bindings.world_storage.protected_count
      if declared:
        registry.columns["counter"] = count
        bindings.fixed_arrays = (count,)
      else:
        bindings.fixed_arrays += (count,)
      with self.subTest(declared=declared), self.assertRaises(ValueError):
        _prepare(fixture)

  def test_scratch_cannot_alias_native_state_or_counters(self):
    for native, temporary in (("nacon", "counter"), ("qpos", "domain0")):
      fixture = _fixture()
      setattr(fixture[1], native, fixture[3].columns[temporary])
      with self.subTest(native=native), self.assertRaisesRegex(ValueError, "overlap"):
        _prepare(fixture)

  def test_mutations_reject_but_device_count_values_can_change(self):
    changes = (
      "source",
      "execution",
      "model",
      "column",
      "declaration",
      "registry_state",
      "registry",
      "bindings",
      "fixed",
      "fixed_wrapper",
      "storage",
      "nested",
      "shallow_dict",
    )
    for change in changes:
      fixture = _fixture()
      model, data, bindings, registry, _ = fixture
      data.nested = {"sequence": [data.qpos]}
      workspace = _prepare(fixture)
      if change == "source":
        data.qpos = wp.zeros(data.qpos.shape, dtype=float, device="cpu")
      elif change == "execution":
        workspace.execution_data.qpos = wp.zeros(data.qpos.shape, dtype=float, device="cpu")
      elif change == "model":
        model.option = 1
      elif change == "column":
        registry.columns["domain0"] = wp.zeros((4, 3), dtype=float, device="cpu")
      elif change == "declaration":
        registry.declarations["domain0"].dtype = wp.int32
      elif change == "registry_state":
        registry.state = scratch_data.FROZEN
      elif change == "registry":
        workspace.execution_data.scratch = dataclasses.replace(registry)
      elif change == "bindings":
        workspace.bindings = dataclasses.replace(bindings)
      elif change == "shallow_dict":
        registry.columns = {**registry.columns, "unexpected": registry.columns["counter"]}
      elif change == "fixed":
        bindings.fixed_arrays = (wp.zeros(1, dtype=int, device="cpu"),)
      elif change == "fixed_wrapper":
        original = registry.columns["counter"]
        alias = wp.array(ptr=original.ptr, shape=original.shape, dtype=original.dtype, device=original.device)
        registry.columns["counter"], bindings.fixed_arrays = alias, (alias,)
      elif change == "storage":
        bindings.world_storage = dataclasses.replace(bindings.world_storage)
      else:
        data.nested["sequence"].append(3)
      with self.subTest(change=change), self.assertRaises(ValueError):
        step_program.validate_step_workspace(workspace, model, data)
    fixture = _fixture()
    workspace = _prepare(fixture)
    fixture[2].world_storage.protected_count.fill_(3)
    fixture[2].contact_storage.ready_count.fill_(4)
    with patch.object(step_program.graph_ops, "current_capture", return_value=None):
      step_program.validate_step_workspace(workspace, fixture[0], fixture[1])

  def test_foreign_model_or_data_and_capture_time_preparation_reject(self):
    fixture = _fixture()
    workspace = _prepare(fixture)
    for model, data in ((dataclasses.replace(fixture[0]), fixture[1]), (fixture[0], dataclasses.replace(fixture[1]))):
      with self.assertRaises(ValueError):
        step_program.validate_step_workspace(workspace, model, data)
    with _cpu_preparation(capturing=True), self.assertRaisesRegex(RuntimeError, "before graph capture"):
      native_workspace.make_step_workspace(
        fixture[0], fixture[1], bindings=fixture[2], scratch=fixture[3], count_parameters=fixture[4]
      )

  def test_retired_or_malformed_domain_counts_reject(self):
    for domain in ("world_storage", "contact_storage", "ccd_storage"):
      for flag in ("closed", "service_failed"):
        fixture = _fixture()
        setattr(getattr(fixture[2], domain), flag, True)
        with self.subTest(domain=domain, flag=flag), self.assertRaises(RuntimeError):
          _prepare(fixture)
    for count in (wp.zeros(2, dtype=int, device="cpu"), wp.zeros(1, dtype=wp.int64, device="cpu")):
      fixture = _fixture()
      fixture[2].ccd_storage.ready_count = count
      with self.assertRaises(ValueError):
        _prepare(fixture)
    for name, value in (("capacity", 0), ("capacity", 2**31), ("capacity", True), ("device", object())):
      fixture = _fixture()
      setattr(fixture[2].contact_storage, name, value)
      with self.subTest(name=name), self.assertRaises(ValueError):
        _prepare(fixture)
    fixture = _fixture()
    fixture[2].world_storage = object()
    with self.assertRaises(TypeError):
      _prepare(fixture)

  def test_updater_uses_world_protection_and_is_required_before_recording(self):
    fixture = _fixture()
    workspace = _prepare(fixture)
    with _capture(fixture[2]) as graph:
      fixture[2].updates = None
      with self.assertRaisesRegex(RuntimeError, "before recording"):
        step_program.validate_step_workspace(workspace, fixture[0], fixture[1])
      self.assertTrue(fixture[2].recording_failed)
      self.assertTrue(graph._preparation_failed)
    for name, value in (("enable_count", "ready"), ("enable_count_maximum", 3), ("device", object())):
      fixture = _fixture()
      bindings = fixture[2]
      bindings.updates = _updates(bindings)
      setattr(bindings.updates, name, bindings.world_storage.ready_count if value == "ready" else value)
      with self.subTest(name=name), self.assertRaisesRegex(ValueError, "exact world protected count"):
        _prepare(fixture)

  def test_recorded_source_and_updater_identity_remain_fixed(self):
    bindings = _fixture()[2]
    with _capture(bindings) as graph:
      step_program._retain_bindings(bindings, graph)
      for owner, name, value in (
        (bindings, "updates", _updates(bindings)),
        (bindings.contact_storage, "ready_count", wp.zeros(1, dtype=int, device="cpu")),
        (bindings, "bindings", []),
      ):
        previous = getattr(owner, name)
        try:
          setattr(owner, name, value)
          with self.subTest(name=name), self.assertRaises(ValueError):
            step_program._validate_bindings(bindings)
        finally:
          setattr(owner, name, previous)

  def test_equal_capacity_launches_keep_their_operand_occurrences(self):
    fixture = _fixture()
    workspace = _prepare(fixture)
    world, candidate, ccd = fixture[4]
    kernel = object()
    records = (
      wp.CapturedLaunch(kernel, (4,), 1, 120, (), (), 128, 0, False, ((0, world),), ((1, candidate),)),
      wp.CapturedLaunch(kernel, (4,), 2, 120, (), (), 128, 0, False, ((0, candidate),), ((1, ccd),)),
      wp.CapturedLaunch(kernel, (4,), 3, 120, (), (), 128, 0, False),
    )
    with (
      _capture(fixture[2]) as graph,
      patch.object(wp, "capture_get_memory_operations", return_value=()),
      patch.object(step_program.field_ops, "validate_memory_operations"),
      patch.object(step_program.graph_ops, "adopt_launches", return_value=(object(),) * 3) as adopt,
    ):
      step_program.bind_step_program(workspace, graph, records)
      self.assertEqual(adopt.call_args.args, (fixture[2].updates, graph, records))
      self.assertEqual(adopt.call_args.kwargs["fixed"], (2,))
      for (parameter, _), expected in zip(adopt.call_args.kwargs["count_sources"], fixture[4], strict=True):
        self.assertIs(parameter, expected)
      self.assertEqual(len(fixture[2].bindings), 3)

  def test_failed_memory_validation_or_adoption_prevents_publication(self):
    for problem in ("metadata", "mutable_sequence", "memory", "adoption"):
      fixture = _fixture()
      workspace = _prepare(fixture)
      record = wp.CapturedLaunch(object(), (4,), 1, 120, (), (), 128, 0, False, ((0, fixture[4][0]),))
      records = [record] if problem == "mutable_sequence" else (record,)
      if problem == "metadata":
        fixture[1].option = 1
      with (
        self.subTest(problem=problem),
        _capture(fixture[2]) as graph,
        patch.object(wp, "capture_get_memory_operations", return_value=()),
        patch.object(
          step_program.field_ops,
          "validate_memory_operations",
          side_effect=ValueError("bad memory") if problem == "memory" else None,
        ),
        patch.object(step_program.graph_ops, "adopt_launches", side_effect=ArithmeticError("bad adoption")) as adopt,
      ):
        with self.assertRaises((ValueError, TypeError, ArithmeticError)):
          step_program.bind_step_program(workspace, graph, records)
        self.assertTrue(fixture[2].recording_failed)
        self.assertTrue(graph._preparation_failed)
        self.assertEqual(fixture[2].bindings, [])
        self.assertEqual(adopt.call_count, int(problem == "adoption"))

  def test_memory_validation_receives_only_owned_occurrences_and_exact_operands(self):
    for mode in ("extent", "scalar", "fixed"):
      fixture = _fixture()
      workspace = _prepare(fixture)
      parameter = fixture[4][0]
      extent = ((0, parameter),) if mode == "extent" else ()
      scalar = ((2, parameter),) if mode == "scalar" else ()
      record = wp.CapturedLaunch(object(), (4,), 1, 120, (), (), 128, 0, False, extent, scalar)
      operation = SimpleNamespace(launch=record)
      excluded = SimpleNamespace(launch=dataclasses.replace(record, node_handle=2))
      with (
        self.subTest(mode=mode),
        _capture(fixture[2]) as graph,
        patch.object(wp, "capture_get_memory_operations", return_value=(excluded, operation)),
        patch.object(step_program.field_ops, "validate_memory_operations") as validate,
        patch.object(step_program.graph_ops, "adopt_launches", return_value=(object(),)) as adopt,
      ):
        step_program.bind_step_program(workspace, graph, (record,))
        self.assertEqual(adopt.call_args.kwargs["fixed"], (0,) if mode == "fixed" else ())
        self.assertIs(validate.call_args.args[0], graph)
        self.assertEqual(validate.call_args.args[1], (operation,))
        self.assertEqual(
          [id(storage) for storage in validate.call_args.args[2]],
          [id(fixture[2].world_storage), id(fixture[2].contact_storage), id(fixture[2].ccd_storage)],
        )
        self.assertEqual(
          {id(array) for array in validate.call_args.kwargs["fixed_arrays"]},
          {id(fixture[1].nacon), id(fixture[1].ncollision), id(fixture[3].columns["counter"])},
        )
        self.assertIs(validate.call_args.kwargs["count_sources"][0][0], parameter)

  def test_numerical_modules_depend_only_on_the_scratch_operation(self):
    for name in ("forward", "solver", "collision_driver", "collision_convex", "constraint", "island", "smooth"):
      path = Path(native_workspace.__file__).with_name(name + ".py")
      for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom):
          module = node.module or ""
          with self.subTest(module=name):
            self.assertFalse(module.endswith((".workspace", ".step_program")))
            if module == "mujoco_warp._src":
              self.assertFalse({alias.name for alias in node.names} & {"workspace", "step_program"})
            if module.startswith("gpu_components"):
              self.assertEqual(module, "gpu_components.scratch")
              self.assertEqual([item.name for item in node.names], ["scratch_array"])


class NumericalBoundaryTest(unittest.TestCase):
  def test_solver_initializes_condition_each_call_from_the_data_count(self):
    registry = scratch_ops.allocate_registry("cpu")
    condition = scratch_ops.scratch_array(registry, "solver.nsolving", (1,), int)
    scratch_ops.freeze(registry)
    scratch_ops.prepare(registry, {"solver.nsolving": condition})
    model = Mock(opt=SimpleNamespace(disableflags=0, solver=types.SolverType.NEWTON, iterations=1, graph_conditional=True))
    with (
      patch.object(wp, "launch"),
      patch.object(solver, "init_context"),
      patch.object(solver, "_use_incremental", return_value=False),
    ):
      for live in (0, 3, 7):
        condition.fill_(-91)
        with patch.object(wp, "capture_while") as loop:
          solver._solve(model, Mock(nworld=live, scratch=registry), Mock())
        np.testing.assert_array_equal(condition.numpy(), [live])
        self.assertIs(loop.call_args.args[0], condition)

  def test_disabled_derivative_preserves_ordinary_copy_semantics(self):
    model = SimpleNamespace(
      has_fluid=False,
      M_fullm_i=None,
      M_fullm_j=None,
      opt=SimpleNamespace(disableflags=types.DisableBit.ACTUATION | types.DisableBit.DAMPER),
    )
    data = SimpleNamespace(nworld=2, M=wp.array(np.arange(12, dtype=np.float32).reshape(4, 3), device="cpu"))
    out = wp.full((4, 3), -7.0, device="cpu")
    with patch.object(wp, "launch", side_effect=AssertionError("Unexpected derivative kernel")):
      derivative.deriv_smooth_vel(model, data, out)
    np.testing.assert_array_equal(out.numpy()[:2], data.M.numpy()[:2])
    np.testing.assert_array_equal(out.numpy()[2:], -7.0)

  def test_disabled_features_initialize_their_outputs_with_ordinary_warp(self):
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
      (forward._energy_pos, 0, 0, ("energy",)),
    )
    for operation, nactuator, disableflags, names in cases:
      with self.subTest(operation=operation.__name__, nactuator=nactuator):
        arrays = {name: wp.full((4, 3), 7.0, device="cpu") for name in names}
        model = SimpleNamespace(nactuator=nactuator, opt=SimpleNamespace(disableflags=disableflags, enableflags=0))
        with patch.object(wp, "launch", side_effect=AssertionError("Unexpected stage work")):
          operation(model, SimpleNamespace(nworld=2, **arrays))
        for array in arrays.values():
          np.testing.assert_array_equal(array.numpy()[:2], 0)
          np.testing.assert_array_equal(array.numpy()[2:], 7.0)

  def test_kinematics_emits_plain_warp_pose_launches(self):
    # Sites are allowed in prepared models, even though keyboard fixtures have none.
    model = Mock(nbranch=2, nbody=3, ngeom=4, nsite=2)
    data = Mock(nworld=7)
    with (
      patch.object(step_program, "validate_step_workspace", side_effect=AssertionError("Unowned workspace validation")),
      patch.object(wp, "launch") as launch,
    ):
      smooth.kinematics(model, data)
    self.assertEqual(launch.call_count, 5)
    for call in launch.call_args_list:
      self.assertNotIn("extent_domain", call.kwargs)
      self.assertNotIn("bindings", call.kwargs)
    self.assertIs(launch.call_args.args[0], smooth._site_local_to_global)
    self.assertEqual(launch.call_args.kwargs["dim"], (7, 2))

  def test_execution_symbols_cannot_be_used_as_allocation_or_readback_metadata(self):
    count = wp.CountParameter(17)
    execution = SimpleNamespace(nworld=count, naconmax=17, naccdmax=17)
    with patch.object(wp, "empty", side_effect=AssertionError("Allocation preceded rejection")):
      for operation in (native_io.make_data, native_io.put_data):
        for name in ("nworld", "nconmax", "nccdmax", "njmax", "njmax_nnz", "naconmax", "naccdmax", "nvmax"):
          args = (None,) if operation is native_io.make_data else (None, None)
          with self.subTest(operation=operation.__name__, count=name), self.assertRaisesRegex(ValueError, "concrete"):
            operation(*args, **{name: count})
      with self.assertRaisesRegex(ValueError, "concrete"):
        native_io.replicate_data(execution, 3)
      with self.assertRaisesRegex(ValueError, "concrete"):
        native_io.get_data_into(None, None, execution)

  def test_same_factory_name_does_not_merge_memoized_kernels(self):
    def same_name(_):
      return smooth._kinematics_branch

    first = warp_util.cache_kernel(same_name)

    def same_name(_):
      return sleep._wake_collision_kernel

    second = warp_util.cache_kernel(same_name)
    self.assertIs(first(7), smooth._kinematics_branch)
    self.assertIs(second(7), sleep._wake_collision_kernel)

  def test_clearing_memoization_releases_cached_objects(self):
    class Kernel:
      pass

    @warp_util.cache_kernel
    def factory(_):
      return Kernel()

    old = factory(7)
    reference = weakref.ref(old)
    self.assertIs(factory(7), old)
    warp_util.clear_kernel_cache()
    new = factory(7)
    self.assertIsNot(old, new)
    del old
    self.assertIsNone(reference())


class ModelAdmissionTest(unittest.TestCase):
  def test_model_contract_and_optional_features_on_real_cpu_metadata(self):
    xml = (
      '<mujoco><option solver="Newton" integrator="implicitfast" jacobian="sparse" cone="pyramidal">'
      '<flag sleep="enable" multiccd="disable"/></option><worldbody><body><joint type="slide"/>'
      '<geom type="sphere" size=".02" mass="1"/></body></worldbody></mujoco>'
    )
    with wp.ScopedDevice("cpu"):
      model = mujoco.MjModel.from_xml_string(xml)
      m, d = mjw.put_model(model), mjw.make_data(model, nworld=2, nconmax=8, nccdmax=8, njmax=8)
    m.opt.broadphase, m.opt.graph_conditional = types.BroadphaseType.NXN, True
    native_workspace.validate_step_model(m, d)
    for owner, name, value in (
      (m, "has_fluid", True),
      (m, "has_sdf_geom", True),
      (m, "ntree", 0),
      (m.opt, "integrator", types.IntegratorType.EULER),
      (m.opt, "solver", types.SolverType.CG),
      (m.opt, "enableflags", m.opt.enableflags | types.EnableBit.ENERGY),
      (m, "nsensor", 1),
      (m.opt, "run_rne_postconstraint", True),
      (d, "nvmax", m.nv + 1),
    ):
      previous = getattr(owner, name)
      try:
        setattr(owner, name, value)
        with self.subTest(name=name), self.assertRaises((NotImplementedError, ValueError)):
          native_workspace.validate_step_model(m, d)
      finally:
        setattr(owner, name, previous)

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
          native_workspace.validate_step_model(model, data)
    # This exact mask bypasses both specialized launches in forward.implicit.
    model.opt.disableflags = base | flags[0] | flags[1] | flags[2]
    self.assertIsNone(native_workspace.validate_step_model(model, data))

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
          self.assertIsNone(native_workspace.validate_step_model(model, data))
          continue
        with (
          patch.object(wp, "empty", side_effect=AssertionError("Unexpected allocation")),
          patch.object(solver, "_compact_solver_views", side_effect=AssertionError("Unexpected scratch planning")),
          self.assertRaisesRegex(NotImplementedError, message),
        ):
          native_workspace.validate_step_model(model, data)
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
        self.assertIsNone(native_workspace.validate_step_model(model, data))

  def test_blocked_matrix_alignment_preserves_each_axis_requirement(self):
    array = wp.empty((2, 8, 8), dtype=float, device="cpu")
    solver.validate_blocked_matrix(array)
    for offset, strides in ((4, (256, 32, 4)), (0, (260, 32, 4)), (0, (256, 36, 4))):
      alias = wp.array(ptr=array.ptr + offset, shape=(1, 2, 2), strides=strides, dtype=float, device="cpu")
      with self.subTest(offset=offset, strides=strides), self.assertRaises(ValueError):
        solver.validate_blocked_matrix(alias)


if __name__ == "__main__":
  unittest.main()
