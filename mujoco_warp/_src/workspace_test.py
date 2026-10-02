"""CPU preparation gates for immutable metadata and unsupported native features."""

import ast
import dataclasses
import inspect
import typing
import unittest
import weakref
from contextlib import ExitStack
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from unittest.mock import PropertyMock
from unittest.mock import patch

import numpy as np
import warp as wp
from gpu_components.field_data import FieldStorage
from gpu_components.graph_data import GraphUpdateTable

from mujoco_warp import test_data
from mujoco_warp._src import collision_convex
from mujoco_warp._src import collision_primitive
from mujoco_warp._src import constraint
from mujoco_warp._src import derivative
from mujoco_warp._src import forward
from mujoco_warp._src import island
from mujoco_warp._src import passive
from mujoco_warp._src import sleep
from mujoco_warp._src import smooth
from mujoco_warp._src import solver
from mujoco_warp._src import step_execution as native_execution
from mujoco_warp._src import step_program
from mujoco_warp._src import support
from mujoco_warp._src import types
from mujoco_warp._src import warp_util
from mujoco_warp._src import workspace as native_workspace
from mujoco_warp._src.step_execution import StepBindings
from mujoco_warp._src.step_execution import resolve_step_counts
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


def _convex_model(pair=(types.GeomType.BOX, types.GeomType.BOX), *, multiccd=False):
  counts = [0] * (len(types.GeomType) * (len(types.GeomType) + 1) // 2)
  counts[collision_convex.upper_trid_index(len(types.GeomType), *[geom.value for geom in pair])] = 1
  return SimpleNamespace(
    geom_pair_type_count=counts,
    npolygonmax=7,
    nmeshdegmax=5,
    opt=SimpleNamespace(ccd_iterations=35, disableflags=0 if multiccd else types.DisableBit.MULTICCD),
  )


def _collision_scratch_specs():
  pair = (types.GeomType.BOX, types.GeomType.BOX)
  shapes = collision_convex._convex_scratch_shapes(_convex_model(pair), [pair], 2)[2]
  hints = typing.get_type_hints(collision_convex._ConvexScratch)
  return (
    WorkspaceFieldSpec("collision_pair", (2,), wp.vec2i, "candidate"),
    WorkspaceFieldSpec("collision_pairid", (2,), wp.vec2i, "candidate"),
    WorkspaceFieldSpec("collision_worldid", (2,), wp.int32, "candidate"),
    *(
      WorkspaceFieldSpec(name, shape, hints[name].dtype, "global_counter" if name == "nccd" else "ccd")
      for name, shape in shapes.items()
    ),
  )


@contextmanager
def _cpu_workspace_preparation(specs):
  """Exercise real preparation arithmetic, replacing only CUDA admission and topology discovery."""
  with (
    patch.object(type(wp.get_device("cpu")), "is_cuda", new_callable=PropertyMock, return_value=True),
    patch.object(wp, "get_stream", return_value=SimpleNamespace(is_capturing=False)),
    patch.object(native_workspace, "step_workspace_layout", return_value=specs),
    patch.object(solver, "_compact_solver_views", side_effect=lambda model, data: (model, data)),
    patch.object(
      solver,
      "_solver_context_layout",
      return_value=tuple(
        (spec.name.removeprefix("solver."), spec.shape, spec.dtype, False) for spec in specs if spec.name.startswith("solver.")
      ),
    ),
    patch.object(types, "SolverContext", side_effect=lambda **fields: SimpleNamespace(**fields)),
  ):
    yield


def _workspace_data(qpos):
  """Supply absent compact-matrix descriptors for CPU preparation fixtures."""
  empty = wp.empty((1, 0, 0), dtype=wp.float32, device="cpu")
  return SimpleNamespace(qpos=qpos, cM=empty, cqLD=empty, nworld=17, naconmax=17, naccdmax=17)


def _step_bindings():
  stores = []
  for _ in range(3):
    protected = wp.zeros(1, dtype=wp.int32, device="cpu")
    stores.append(
      FieldStorage(
        17,
        protected,
        protected.device,
        None,
        0,
        {},
        {},
        {},
        {},
        {},
        weakref.WeakSet(),
        weakref.WeakSet(),
        ready_count=wp.zeros(1, dtype=wp.int32, device="cpu"),
      )
    )
  return StepBindings(*stores)


def _update_table(bindings):
  count = bindings.world_storage.protected_count
  return GraphUpdateTable(count, 17, count.device, 32, 120, count, count, count, None, {})


def _native_kernel():
  return SimpleNamespace(
    key="named_counts",
    func=SimpleNamespace(__module__="test", __qualname__="named_counts"),
    adj=SimpleNamespace(
      kernel_dim=2,
      args=[SimpleNamespace(label=name, type=wp.int32) for name in ("unrelated", "world_live_count", "contact_cap", "ccd_cap")],
    ),
  )


@contextmanager
def _binding_capture(bindings):
  bindings.updates = _update_table(bindings)

  class CapturedGraph:
    pass

  graph = CapturedGraph()
  graph.device = bindings.world_storage.device

  with (
    patch.object(native_execution.graph_ops, "current_capture", return_value=graph),
    patch.object(wp, "get_stream", return_value=None),
  ):
    yield graph


class NativeBindingsTest(unittest.TestCase):
  def test_physics_launches_stay_plain_and_prepared_signatures_have_resource_purpose(self):
    """Gate the native execution boundary independently of numerical output parity."""
    for path in Path(native_execution.__file__).parent.glob("*.py"):
      if path.name.endswith("_test.py"):
        continue
      with self.subTest(module=path.name):
        tree = ast.parse(path.read_text())
        self.assertFalse(
          any(
            isinstance(node, (ast.FunctionDef, ast.Name, ast.Attribute))
            and getattr(node, "name", getattr(node, "id", getattr(node, "attr", None)))
            in ("bind_step_launch", "launch_step_kernel")
            for node in ast.walk(tree)
          )
        )
    import mujoco_warp as mjw

    for name in (
      "euler",
      "rungekutta4",
      "rne_postconstraint",
      "energy_vel",
      "sensor_pos",
      "sensor_vel",
      "sensor_acc",
    ):
      with self.subTest(operation=name):
        self.assertNotIn("workspace", inspect.signature(getattr(mjw, name)).parameters)
    for operation in (
      sleep.update_sleep,
      sleep.wake,
      sleep.wake_collision,
      island.update_active_dofs,
      smooth.factor_solve_i,
      support.apply_ft,
      collision_convex.convex_narrowphase,
      collision_primitive.primitive_narrowphase,
      smooth.kinematics,
      smooth.com_pos,
      smooth.com_vel,
      support.mul_m,
      support.xfrc_accumulate,
      forward.fwd_kinematics,
      forward.fwd_acceleration,
      solver._mul_m_compact_aware,
      solver._compact_scatter,
      solver._linesearch,
      solver._solver_iteration,
    ):
      with self.subTest(operation=operation.__name__):
        self.assertFalse({"workspace", "bindings"} & inspect.signature(operation).parameters.keys())

  def test_public_binding_record_is_passive_and_package_resources_are_canonical(self):
    import mujoco_warp as mjw

    self.assertIs(mjw.StepBindings, StepBindings)
    self.assertFalse(hasattr(mjw, "validate_step_launch"))
    self.assertFalse(hasattr(mjw, "_resolve_launch_counts"))
    self.assertFalse(hasattr(mjw, "bind_step_launch"))
    self.assertFalse(hasattr(native_execution, "bind_step_launch"))
    self.assertFalse(hasattr(native_execution, "_record_launch"))
    self.assertFalse(hasattr(mjw, "launch_step_kernel"))
    self.assertIs(mjw.resolve_step_counts, resolve_step_counts)
    self.assertIs(mjw.bind_step_program, step_program.bind_step_program)
    self.assertEqual(
      [f.name for f in dataclasses.fields(StepBindings) if not f.name.startswith("_")],
      [
        "world_storage",
        "contact_storage",
        "ccd_storage",
        "updates",
        "recording_failed",
        "bindings",
      ],
    )
    tree = ast.parse(inspect.getsource(StepBindings))
    self.assertFalse(any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in ast.walk(tree)))
    first = _step_bindings()
    second = dataclasses.replace(first)
    self.assertNotEqual(first, second)
    self.assertIs(weakref.ref(first)(), first)
    self.assertIs(StepBindings.__annotations__["world_storage"], FieldStorage)
    self.assertEqual(StepBindings.__annotations__["updates"], GraphUpdateTable | None)
    self.assertFalse(any(name in inspect.getsource(native_execution) for name in ("import newton", "recorder=")))

  def test_numerical_helpers_borrow_only_resources_they_use(self):
    for operation in (
      solver._linesearch_iterative,
      solver._update_constraint,
      solver._cholesky_factorize_solve,
      solver._compact_gather,
      forward._energy_pos,
      smooth._rne_cacc_world,
      smooth._rne_cacc_forward,
      smooth._rne_cfrc,
      smooth._rne_cfrc_backward,
      smooth._factor_solve_blocks,
    ):
      with self.subTest(operation=operation.__name__):
        parameters = inspect.signature(operation).parameters
        needs_bounded_memory = operation in (solver._compact_gather, forward._energy_pos, smooth._rne_cacc_world)
        self.assertEqual("bindings" in parameters, needs_bounded_memory)
        self.assertNotIn("workspace", parameters)
        self.assertFalse(
          any(
            isinstance(node, ast.Name) and node.id == "workspace" for node in ast.walk(ast.parse(inspect.getsource(operation)))
          )
        )
    for module in (solver, passive, island, forward, smooth, sleep):
      for node in ast.walk(ast.parse(inspect.getsource(module))):
        if not isinstance(node, ast.If) or not node.orelse:
          continue
        eager_calls = [item.value for item in node.body if isinstance(item, ast.Expr) and isinstance(item.value, ast.Call)]
        bounded_calls = [item.value for item in node.orelse if isinstance(item, ast.Expr) and isinstance(item.value, ast.Call)]
        if len(eager_calls) != len(node.body) or len(bounded_calls) != len(node.orelse):
          continue
        duplicate = all(
          isinstance(call.func, ast.Attribute) and call.func.attr in ("zero_", "fill_", "copy") for call in eager_calls
        ) and all(
          ast.unparse(call.func) in ("step_execution.fill_step_rows", "step_execution.copy_step_rows") for call in bounded_calls
        )
        self.assertFalse(duplicate, f"{module.__name__}:{node.lineno}: memory operations already support eager bindings")

  def test_execution_dependencies_and_exports_have_one_canonical_owner(self):
    self.assertEqual(StepBindings.__module__, native_execution.__name__)
    tree = ast.parse(inspect.getsource(native_execution))
    for node in ast.walk(tree):
      if isinstance(node, ast.Import):
        imports = [alias.name for alias in node.names]
      elif isinstance(node, ast.ImportFrom):
        imports = [node.module or ""]
        self.assertEqual(node.level, 0, "Native execution must not depend on neighboring physics modules")
      else:
        continue
      self.assertFalse(
        any(name == "mujoco_warp" or name.startswith("mujoco_warp.") for name in imports),
        "Execution may depend on generic components and Warp, never workspace or physics stages",
      )
    for name in (
      "StepBindings",
      "resolve_step_counts",
      "_binding_layout",
      "_validate_bindings",
      "_begin_recording",
      "_fail_recording",
      "fill_step_rows",
      "copy_step_rows",
      "validate_step_workspace",
      "_layout",
    ):
      self.assertNotIn(
        name, vars(native_workspace), "Preparation must call the execution module without aliases or duplicate owners"
      )
    convex_tree = ast.parse(inspect.getsource(collision_convex))
    imports = set()
    for node in ast.walk(convex_tree):
      if isinstance(node, ast.Import):
        imports.update(alias.name for alias in node.names)
      elif isinstance(node, ast.ImportFrom):
        imports.add(node.module)
        imports.update(f"{node.module}.{alias.name}" for alias in node.names)
    self.assertNotIn("mujoco_warp._src.workspace", imports)
    self.assertNotIn("mujoco_warp._src.step_execution", imports)

  def test_invalid_declarations_preflight_without_emission_or_poisoning(self):
    cases = [
      ("unknown", 0, {}),
      (None, 0, {}),
      ("world", None, {}),
      ("world", 1, {}),
      ("world", False, {}),
      ("world", 0.0, {}),
      ("world", 0, {"missing": "world"}),
      ("world", 0, {"world_live_count": "unknown"}),
      ("world", 0, {str(index): "world" for index in range(5)}),
    ]
    bindings, kernel = _step_bindings(), _native_kernel()
    with patch.object(native_execution.graph_ops, "register_last_kernel_node", side_effect=AssertionError("node claimed")):
      for domain, axis, parameters in cases:
        with self.subTest(domain=domain, axis=axis, parameters=parameters), self.assertRaises(ValueError):
          resolve_step_counts(bindings, kernel, domain, axis, parameters)
      kernel.adj.args[1].type = wp.int64
      with self.assertRaisesRegex(ValueError, "int32"):
        resolve_step_counts(bindings, kernel, "world", 0, {"world_live_count": "world"})
    self.assertFalse(bindings.recording_failed)
    self.assertEqual(bindings.bindings, [])
    self.assertIsNone(bindings._recording_binding)

  def test_bounded_memory_operations_use_explicit_native_domain_sources(self):
    bindings = _step_bindings()
    array, source = (wp.empty(12, dtype=wp.float32, device="cpu") for _ in range(2))
    with (
      _binding_capture(bindings),
      patch.object(native_execution.field_ops, "fill") as fill,
      patch.object(native_execution.field_ops, "copy") as copy,
      patch.object(native_execution.field_ops, "lookup", return_value=SimpleNamespace(name="field")),
    ):
      self.assertIs(native_execution.fill_step_rows(bindings, array, 1, "world"), array)
      native_execution.fill_step_rows(bindings, array, 0, "candidate")
      native_execution.copy_step_rows(bindings, array, source, "ccd")
      self.assertIs(fill.call_args_list[0].args[0], bindings.world_storage)
      self.assertIs(fill.call_args_list[0].kwargs["count"], bindings.world_storage.protected_count)
      self.assertIs(fill.call_args_list[1].kwargs["count"], bindings.contact_storage.ready_count)
      self.assertIs(copy.call_args.kwargs["count"], bindings.ccd_storage.ready_count)
      empty = wp.empty(0, dtype=wp.float32, device="cpu")
      native_execution.fill_step_rows(bindings, empty, 0, "world")
      native_execution.copy_step_rows(bindings, empty, empty, "world")
      self.assertEqual(fill.call_count, 2)
      self.assertEqual(copy.call_count, 1)

  def test_eager_zero_uses_memset_path_and_negative_zero_keeps_its_bits(self):
    array = wp.full(4, 7.0, dtype=wp.float32, device="cpu")
    with patch.object(array, "fill_", side_effect=AssertionError("Zeroing must keep Warp's zero_ fast path")):
      self.assertIs(native_execution.fill_step_rows(None, array, 0, "world"), array)
    np.testing.assert_array_equal(array.numpy().view(np.uint32), np.zeros(4, dtype=np.uint32))
    native_execution.fill_step_rows(None, array, -0.0, "world")
    np.testing.assert_array_equal(array.numpy().view(np.uint32), np.full(4, 0x80000000, dtype=np.uint32))

  def test_empty_memory_operations_validate_owner_domain_and_descriptor_without_recording(self):
    """An empty payload is not an escape from ownership admission or descriptor compatibility."""
    empty = wp.empty(0, dtype=wp.float32, device="cpu")
    nonleading_empty = wp.empty((17, 0), dtype=wp.float32, device="cpu")
    bindings = _step_bindings()
    with (
      patch.object(native_execution, "_begin_recording", side_effect=AssertionError("Empty operation recorded")),
      patch.object(native_execution.field_ops, "fill", side_effect=AssertionError("Empty fill emitted")),
      patch.object(native_execution.field_ops, "copy", side_effect=AssertionError("Empty copy emitted")),
    ):
      self.assertIs(native_execution.fill_step_rows(bindings, empty, 0, "world"), empty)
      native_execution.copy_step_rows(bindings, empty, empty, "world")
      self.assertIsNone(bindings._recording_binding)
      for operation in ("fill", "copy"):
        for problem in ("closed", "quarantined", "domain", "unregistered", "source"):
          if problem == "source" and operation == "fill":
            continue
          bindings = _step_bindings()
          bindings.world_storage.closed = problem == "closed"
          bindings.world_storage.service_failed = problem == "quarantined"
          array = nonleading_empty if problem == "unregistered" else empty
          source = wp.empty(0, dtype=wp.int32, device="cpu") if problem == "source" else array
          domain = "missing" if problem == "domain" else "world"
          with self.subTest(operation=operation, problem=problem), self.assertRaises((ValueError, RuntimeError)):
            if operation == "fill":
              native_execution.fill_step_rows(bindings, array, 0, domain)
            else:
              native_execution.copy_step_rows(bindings, array, source, domain)

  def test_post_emission_failures_poison_native_and_shared_graph_without_fallback(self):
    for name in ("fill", "copy"):
      for error in (ArithmeticError, KeyboardInterrupt):
        bindings = _step_bindings()
        array = wp.empty(12, dtype=wp.float32, device="cpu")
        operation = {
          "fill": lambda: native_execution.fill_step_rows(bindings, array, 0, "world"),
          "copy": lambda: native_execution.copy_step_rows(bindings, array, array, "world"),
        }[name]
        with (
          self.subTest(operation=name, error=error),
          _binding_capture(bindings) as graph,
          patch.object(native_execution.field_ops, "fill", side_effect=error("recording failed")),
          patch.object(native_execution.field_ops, "copy", side_effect=error("recording failed")),
          patch.object(wp, "copy", side_effect=AssertionError("Unexpected dense copy")),
        ):
          with self.assertRaisesRegex(error, "recording failed"):
            operation()
          self.assertTrue(bindings.recording_failed)
          self.assertTrue(graph._preparation_failed)
          self.assertTrue(any(owner is bindings for owner in graph._resource_owners))
          with self.assertRaisesRegex(RuntimeError, "failed recording"):
            resolve_step_counts(bindings, _native_kernel(), "world", 0, {})

  def test_recording_rejects_replaced_sources_updater_or_bindings_but_allows_count_values(self):
    bindings, kernel = _step_bindings(), _native_kernel()
    with _binding_capture(bindings):
      with patch.object(native_execution.field_ops, "fill"):
        native_execution.fill_step_rows(bindings, wp.empty(1, device="cpu"), 0, "world")
      for owner, name, value in (
        (bindings, "updates", _update_table(bindings)),
        (bindings, "world_storage", dataclasses.replace(bindings.world_storage)),
        (bindings.contact_storage, "ready_count", wp.zeros(1, dtype=wp.int32, device="cpu")),
        (bindings.ccd_storage, "capacity", 18),
        (bindings, "bindings", []),
      ):
        original = getattr(owner, name)
        try:
          setattr(owner, name, value)
          with self.subTest(name=name), self.assertRaisesRegex(ValueError, "recording binding changed"):
            resolve_step_counts(bindings, kernel, "world", 0, {})
        finally:
          setattr(owner, name, original)
      bindings.world_storage.protected_count.fill_(3)
      bindings.contact_storage.ready_count.fill_(5)
      resolve_step_counts(bindings, kernel, "world", 0, {})
      self.assertFalse(bindings.recording_failed)

  def test_sources_and_update_enable_count_have_one_bounded_device_contract(self):
    for name, value in (("capacity", 0), ("capacity", 2**31), ("capacity", True), ("device", object())):
      bindings = _step_bindings()
      setattr(bindings.contact_storage, name, value)
      with self.subTest(name=name), self.assertRaises(ValueError):
        resolve_step_counts(bindings, _native_kernel(), "world", 0, {})
    for count in (wp.zeros(2, dtype=wp.int32, device="cpu"), wp.zeros(1, dtype=wp.int64, device="cpu")):
      bindings = _step_bindings()
      bindings.ccd_storage.ready_count = count
      with self.assertRaisesRegex(ValueError, "int32 scalars"):
        resolve_step_counts(bindings, _native_kernel(), "world", 0, {})
    bindings = _step_bindings()
    bindings.updates = _update_table(bindings)
    bindings.updates.enable_count = bindings.world_storage.ready_count
    with self.assertRaisesRegex(ValueError, "exact world protected count"):
      resolve_step_counts(bindings, _native_kernel(), "world", 0, {})

  def test_retired_storage_rejects_before_emission_and_malformed_borrows_preserve_errors(self):
    for domain in ("world_storage", "contact_storage", "ccd_storage"):
      for flag in ("closed", "service_failed"):
        bindings = _step_bindings()
        setattr(getattr(bindings, domain), flag, True)
        with self.subTest(domain=domain, flag=flag), self.assertRaisesRegex(RuntimeError, "closed or quarantined"):
          resolve_step_counts(bindings, _native_kernel(), "world", 0, {})
        self.assertFalse(bindings.recording_failed)
    bindings = _step_bindings()
    bindings.world_storage = object()
    with self.assertRaisesRegex(TypeError, "borrow FieldStorage"):
      resolve_step_counts(bindings, _native_kernel(), "world", 0, {})
    self.assertFalse(bindings.recording_failed)
    bindings = _step_bindings()
    with _binding_capture(bindings) as graph:
      with patch.object(native_execution.field_ops, "fill"):
        native_execution.fill_step_rows(bindings, wp.empty(1, device="cpu"), 0, "world")
      bindings.world_storage = object()
      with self.assertRaisesRegex(TypeError, "borrow FieldStorage"):
        resolve_step_counts(bindings, _native_kernel(), "world", 0, {})
      self.assertFalse(getattr(graph, "_preparation_failed", False))


class StepProgramTest(unittest.TestCase):
  def test_native_domains_resolve_independent_sources_at_composition(self):
    bindings = _step_bindings()
    bindings.contact_storage.capacity, bindings.ccd_storage.capacity = 19, 23
    ccd = collision_convex.ccd_kernel_builder(6, 6, 35, 16, False, 0, 32, 0)
    kernels = (smooth._kinematics_branch, sleep._wake_collision_kernel, ccd, forward._next_time_builder(0))
    records = tuple(
      wp.CapturedLaunch(kernel, dim, index + 1, 120, (), (), 128, 0, False)
      for index, (kernel, dim) in enumerate(zip(kernels, ((17, 4), (19,), (8192,), (17,))))
    )
    # Preparation may regenerate future factories after a module option change.
    # The captured records must still identify their old, retained kernels.
    warp_util.clear_kernel_cache()
    with (
      _binding_capture(bindings) as graph,
      patch.object(wp, "capture_get_launches", return_value=records),
      patch.object(step_program.field_ops, "validate_captured_operation", return_value=False),
      patch.object(step_program.graph_ops, "adopt_launch", side_effect=[object() for _ in records]) as adopt,
    ):
      step_program.bind_step_program(bindings, graph, records)
      self.assertEqual(len(bindings.bindings), 4)
      calls = adopt.call_args_list
      for call, record in zip(calls, records):
        self.assertIs(call.args[0], bindings.updates)
        self.assertIs(call.args[1], graph)
        self.assertIs(call.args[2], record)
      self.assertIs(calls[0].kwargs["extent_source"], bindings.world_storage.protected_count)
      self.assertIs(calls[1].kwargs["extent_source"], bindings.contact_storage.ready_count)
      self.assertIsNone(calls[2].kwargs["extent_source"])
      self.assertIsNone(calls[2].kwargs["extent_axis"])
      for call, expected in (
        (calls[2], (("naconmax_in", bindings.contact_storage), ("naccdmax_in", bindings.ccd_storage))),
        (calls[3], (("nworld_in", bindings.world_storage), ("naconmax_in", bindings.contact_storage))),
      ):
        labels = [argument.label for argument in call.args[2].kernel.adj.args]
        for parameter, (label, storage) in zip(call.kwargs["parameters"], expected):
          self.assertEqual(parameter.argument_index, labels.index(label) + 1)
          count = storage.protected_count if storage is bindings.world_storage else storage.ready_count
          self.assertIs(parameter.source, count)
          self.assertEqual(parameter.maximum, storage.capacity)
      self.assertEqual(sum(owner is bindings for owner in graph._resource_owners), 1)
      for storage in (bindings.world_storage, bindings.contact_storage, bindings.ccd_storage):
        self.assertIn(graph, storage._graphs)

  def test_unknown_or_foreign_records_fail_before_publication(self):
    kernel = smooth._kinematics_branch
    record = wp.CapturedLaunch(kernel, (17, 4), 1, 120, (), (), 128, 0, False)
    for problem in ("unknown", "copied", "duplicate", "capacity", "mutable_sequence", "adoption"):
      bindings = _step_bindings()
      launches = (record,)
      if problem == "unknown":
        launches = (dataclasses.replace(record, kernel=Mock(key=kernel.key)),)
      elif problem == "copied":
        launches = (dataclasses.replace(record),)
      elif problem == "duplicate":
        launches = (record, record)
      elif problem == "capacity":
        launches = (dataclasses.replace(record, dim=(16, 4)),)
      elif problem == "mutable_sequence":
        launches = [record]
      captured = (record,) if problem == "copied" else tuple(launches)
      with (
        self.subTest(problem=problem),
        _binding_capture(bindings) as graph,
        patch.object(wp, "capture_get_launches", return_value=captured),
        patch.object(step_program.field_ops, "validate_captured_operation", return_value=False),
        patch.object(step_program.graph_ops, "adopt_launch", side_effect=ArithmeticError("failed adoption")) as adopt,
      ):
        with self.assertRaises((ValueError, TypeError, ArithmeticError)):
          step_program.bind_step_program(bindings, graph, launches)
        self.assertTrue(bindings.recording_failed)
        self.assertTrue(graph._preparation_failed)
        self.assertEqual(bindings.bindings, [])
        self.assertEqual(adopt.call_count, int(problem == "adoption"))

  def test_field_operations_require_validated_storage_and_count_operands(self):
    record = wp.CapturedLaunch(Mock(key="field_operation"), (17,), 1, 120, (), (), 128, 0, False)
    for valid in (False, True):
      bindings = _step_bindings()
      with (
        self.subTest(valid=valid),
        _binding_capture(bindings) as graph,
        patch.object(wp, "capture_get_launches", return_value=(record,)),
        patch.object(
          step_program.field_ops,
          "validate_captured_operation",
          return_value=True,
          side_effect=None if valid else ValueError("foreign count"),
        ) as validate,
        patch.object(step_program.graph_ops, "adopt_launch") as adopt,
      ):
        if valid:
          step_program.bind_step_program(bindings, graph, (record,))
          self.assertFalse(bindings.recording_failed)
        else:
          with self.assertRaisesRegex(ValueError, "foreign count"):
            step_program.bind_step_program(bindings, graph, (record,))
          self.assertTrue(graph._preparation_failed)
        sources = validate.call_args.args[1]
        self.assertEqual(
          [id(owner) for owner, _ in sources],
          [id(bindings.world_storage), id(bindings.contact_storage), id(bindings.ccd_storage)],
        )
        self.assertIs(sources[0][1], bindings.world_storage.protected_count)
        self.assertIs(sources[1][1], bindings.contact_storage.ready_count)
        self.assertIs(sources[2][1], bindings.ccd_storage.ready_count)
        adopt.assert_not_called()

  def test_same_factory_name_does_not_merge_kernel_identities(self):
    def same_name(_):
      return smooth._kinematics_branch

    first = warp_util.cache_kernel(same_name)

    def same_name(_):
      return sleep._wake_collision_kernel

    second = warp_util.cache_kernel(same_name)
    self.assertIs(first(7), smooth._kinematics_branch)
    self.assertIs(second(7), sleep._wake_collision_kernel)
    self.assertEqual(warp_util.kernel_instances(first), (smooth._kinematics_branch,))
    self.assertEqual(warp_util.kernel_instances(second), (sleep._wake_collision_kernel,))
    with self.assertRaisesRegex(ValueError, "exact cache_kernel factory"):
      warp_util.kernel_instances(same_name)

  def test_clearing_memoization_preserves_only_live_factory_provenance(self):
    class Kernel:
      pass

    @warp_util.cache_kernel
    def factory(_):
      return Kernel()

    old = factory(7)
    reference = weakref.ref(old)
    self.assertIs(factory(7), old)
    warp_util.clear_kernel_cache()
    self.assertEqual(warp_util.kernel_instances(factory), (old,))
    new = factory(7)
    self.assertIsNot(old, new)
    self.assertEqual(set(warp_util.kernel_instances(factory)), {old, new})
    del old
    self.assertIsNone(reference())
    self.assertEqual(warp_util.kernel_instances(factory), (new,))


class ConvexScratchTest(unittest.TestCase):
  def test_mesh_cylinder_layout_keeps_unrestricted_eager_requirements(self):
    pair = (types.GeomType.CYLINDER, types.GeomType.MESH)
    model = _convex_model(pair, multiccd=True)
    with patch.object(wp, "empty", side_effect=AssertionError("Layout allocated")):
      count, iterations, shapes = collision_convex._convex_scratch_shapes(model, [pair], 11)
    hints = typing.get_type_hints(collision_convex._ConvexScratch)
    self.assertEqual((count, iterations), (1, 35))
    self.assertEqual(shapes["multiccd_pdist"], (11, 16))
    self.assertIs(hints["multiccd_pdist"].dtype, wp.float32)
    self.assertEqual(shapes["multiccd_polygon"], (11, 32))
    self.assertEqual(shapes["multiccd_idx1"], (11, 5))
    self.assertEqual(shapes["epa_vert"], (11, 80))
    self.assertIs(hints["epa_vert"].dtype, wp.vec3)
    self.assertIs(hints["nccd"].dtype, wp.int32)
    self.assertEqual(set(shapes), {field.name for field in dataclasses.fields(collision_convex._ConvexScratch)})
    for name, shape in shapes.items():
      self.assertEqual(len(shape), hints[name].ndim)

  def test_box_clipping_remains_enabled_when_multiccd_is_disabled(self):
    pair = (types.GeomType.BOX, types.GeomType.BOX)
    count, iterations, fields = collision_convex._convex_scratch_shapes(_convex_model(pair), [pair], 13)
    self.assertEqual((count, iterations), (1, 16))
    self.assertEqual(fields["epa_vert"], (13, 42))
    self.assertEqual(fields["multiccd_pdist"], (13, 4))
    self.assertEqual(fields["multiccd_idx1"], (13, 3))

  def test_scratch_schema_is_shared_and_stage_has_no_per_field_allocation_fallbacks(self):
    record = ast.parse(inspect.getsource(collision_convex._ConvexScratch))
    self.assertFalse(any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in ast.walk(record)))
    stage = ast.parse(inspect.getsource(collision_convex.convex_narrowphase))
    plan = ast.parse(inspect.getsource(step_workspace_layout))
    self.assertTrue(
      any(isinstance(node, ast.Call) and ast.unparse(node.func) == "_convex_scratch_shapes" for node in ast.walk(plan))
    )
    self.assertFalse(any(isinstance(node, ast.Name) and node.id == "workspace" for node in ast.walk(stage)))
    self.assertFalse(
      any(
        isinstance(node, ast.IfExp)
        and isinstance(node.body, ast.Call)
        and ast.unparse(node.body.func) in ("wp.empty", "wp.zeros")
        for node in ast.walk(stage)
      )
    )
    fields = {field.name for field in dataclasses.fields(collision_convex._ConvexScratch)}
    self.assertFalse(
      (fields - {"nccd"})
      & {node.value for node in ast.walk(plan) if isinstance(node, ast.Constant) and isinstance(node.value, str)},
      "Step preparation must consume the collision schema rather than duplicate its field declarations",
    )

  def test_eager_and_borrowed_scratch_clear_counter_at_each_collision(self):
    pair = (types.GeomType.BOX, types.GeomType.BOX)
    metadata = _convex_model(pair)
    model = Mock(**vars(metadata))
    model.opt.ccd_tolerance, model.opt.warn_overflow = 1e-6, False
    model.block_dim = SimpleNamespace(convex_ccd=32)
    data = Mock(naconmax=2, naccdmax=2, ncollision=wp.zeros(1, dtype=wp.int32, device="cpu"))
    shapes = collision_convex._convex_scratch_shapes(model, [pair], 2)[2]
    hints = typing.get_type_hints(collision_convex._ConvexScratch)
    fields = {name: wp.empty(shape, dtype=hints[name].dtype, device="cpu") for name, shape in shapes.items()}
    scratch = collision_convex._ConvexScratch(**fields)
    allocate = wp.empty
    for prepared in (False, True):
      with (
        self.subTest(prepared=prepared),
        wp.ScopedDevice("cpu"),
        patch.object(collision_convex, "ccd_kernel_builder", return_value=object()),
        patch.object(collision_convex, "_ccd_grid_size", return_value=2),
        patch.object(wp, "launch") as launch,
        patch.object(wp, "empty", side_effect=AssertionError("Borrowed scratch allocated") if prepared else allocate),
      ):
        for _ in range(2):
          fields["nccd"].fill_(9)
          collision_convex.convex_narrowphase(model, data, Mock(), [pair], scratch=scratch if prepared else None)
          arguments = launch.call_args.kwargs["inputs"]
          np.testing.assert_array_equal(arguments[-1].numpy(), np.zeros(fields["nccd"].shape, dtype=np.int32))
          if prepared:
            for field in fields.values():
              self.assertTrue(any(argument is field for argument in arguments))
          else:
            self.assertIsNot(arguments[-1], fields["nccd"])
        self.assertEqual(launch.call_count, 2)


class WorkspaceTest(unittest.TestCase):
  def setUp(self):
    # Validation needs no allocated scratch or native topology. This fixture is passive data.
    model, data = Metadata(), Metadata()
    self.metadata_workspace = _StepWorkspace(
      model=model,
      data=data,
      bindings=None,
      device=wp.get_device("cpu"),
      storage=None,
      arrays={},
      convex=None,
      _collision=None,
      _solver_model=None,
      _solver_data=None,
      _solver_context=None,
      _specs=(),
      _allocation_offsets_bytes=None,
      _execution_binding=None,
      _binding_layout=None,
      _data_layout=native_execution._layout(data),
      _model_layout=native_execution._layout(model),
      _scratch_layout=native_execution._layout(((), None, None, None, None, None)),
    )

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
    self.assertEqual(set(inspect.signature(make_step_workspace).parameters), {"model", "data", "arrays", "bindings"})
    self.assertTrue(dataclasses.is_dataclass(_StepWorkspace))
    record = ast.parse(inspect.getsource(_StepWorkspace))
    self.assertFalse(any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in ast.walk(record)))
    self.assertFalse(
      {"validate", "memory_report", "bind_launch", "fill", "copy", "world_live_count"} & vars(_StepWorkspace).keys()
    )
    self.assertNotIn("_ledger", _StepWorkspace.__dataclass_fields__)
    self.assertIs(mjw.step_workspace_memory_report, native_workspace.step_workspace_memory_report)

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

          def copy(bindings, destination, source, domain):
            if bindings is None:
              wp.copy(destination, source)
              return
            calls.append((destination, source, domain))
            wp.copy(destination[:live], source[:live])

          workspace = SimpleNamespace(bindings=object()) if prepared else None
          with (
            patch.object(wp, "launch", side_effect=AssertionError("Unexpected derivative kernel")),
            patch.object(native_execution, "validate_step_workspace"),
            patch.object(native_execution, "copy_step_rows", side_effect=copy),
          ):
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

            def fill(bindings, array, value, domain):
              if bindings is None:
                array.fill_(value)
                return
              self.assertEqual(domain, "world")
              touched.append(array)
              array[:live].fill_(value)

            workspace = SimpleNamespace(bindings=object()) if prepared else None
            with (
              patch.object(wp, "launch", side_effect=AssertionError("Unexpected stage work")),
              patch.object(native_execution, "validate_step_workspace"),
              patch.object(native_execution, "fill_step_rows", side_effect=fill),
            ):
              if operation is smooth._rne_cacc_world:
                operation(model, data, bindings=None if workspace is None else workspace.bindings)
              else:
                operation(model, data, workspace=workspace)
            expected = np.full((4, 3), 7, np.float32)
            expected[: live if prepared else 4] = 0
            for array in arrays.values():
              np.testing.assert_array_equal(array.numpy(), expected)
            if prepared:
              self.assertEqual([id(array) for array in touched], [id(array) for array in arrays.values()])

  def test_kinematics_emits_plain_warp_pose_launches(self):
    # Sites are allowed in prepared models, even though keyboard fixtures have none.
    model = Mock(nbranch=2, nbody=3, ngeom=4, nsite=2)
    data = Mock(nworld=7)
    with (
      patch.object(native_execution, "validate_step_workspace", side_effect=AssertionError("Unowned workspace validation")),
      patch.object(wp, "launch") as launch,
    ):
      smooth.kinematics(model, data)
    self.assertEqual(launch.call_count, 5)
    for call in launch.call_args_list:
      self.assertNotIn("extent_domain", call.kwargs)
      self.assertNotIn("bindings", call.kwargs)
    self.assertIs(launch.call_args.args[0], smooth._site_local_to_global)
    self.assertEqual(launch.call_args.kwargs["dim"], (7, 2))

  def test_solver_conditional_borrows_count_without_forwarding_workspace(self):
    tree = ast.parse(inspect.getsource(solver._solve))
    calls = [
      node
      for node in ast.walk(tree)
      if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "capture_while"
    ]
    self.assertEqual(len(calls), 1)
    arguments = {keyword.arg: ast.unparse(keyword.value) for keyword in calls[0].keywords}
    self.assertEqual(arguments["while_body"], "_solver_iteration")
    self.assertEqual(arguments["nsolving"], "nsolving")
    self.assertNotIn("workspace", arguments)
    count_source = "workspace.bindings.world_storage.protected_count"
    self.assertTrue(
      any(
        isinstance(node, ast.Call)
        and ast.unparse(node.func) == "wp.copy"
        and [ast.unparse(argument) for argument in node.args] == ["nsolving", count_source]
        for node in ast.walk(tree)
      ),
      "Prepared solver initializes its condition from the live count before entering the captured loop",
    )

  def test_explicit_memory_operations_preserve_eager_behavior_without_bindings(self):
    source = wp.array(np.arange(12, dtype=np.float32).reshape(3, 4), device="cpu")
    destination = wp.zeros((3, 4), dtype=wp.float32, device="cpu")
    native_execution.copy_step_rows(None, destination, source, "world")
    np.testing.assert_array_equal(destination.numpy(), source.numpy())
    self.assertIs(native_execution.fill_step_rows(None, destination, 7, "world"), destination)
    np.testing.assert_array_equal(destination.numpy(), np.full((3, 4), 7, np.float32))

  def test_callback_recorder_rejected_before_scratch_planning(self):
    data = SimpleNamespace(qpos=SimpleNamespace(device=SimpleNamespace(is_cuda=True)))
    recorder = SimpleNamespace(bind_launch=lambda *_: None, fill=lambda *_: None, copy=lambda *_: None)
    with (
      patch.object(wp, "get_stream", return_value=SimpleNamespace(is_capturing=False)),
      patch.object(native_workspace, "step_workspace_layout", side_effect=AssertionError("Unexpected scratch planning")),
      self.assertRaisesRegex(TypeError, "StepBindings"),
    ):
      make_step_workspace(None, data, arrays={}, bindings=recorder)
    for arguments in ({"recorder": recorder}, {"world_live_count": object()}):
      with self.subTest(arguments=arguments), self.assertRaisesRegex(TypeError, "unexpected keyword"):
        make_step_workspace(None, data, **arguments)

  def test_native_domain_capacity_mismatch_rejects_before_scratch_planning(self):
    device = SimpleNamespace(is_cuda=True)
    data = SimpleNamespace(qpos=SimpleNamespace(device=device), nworld=17, naconmax=17, naccdmax=17)
    bindings = _step_bindings()
    bindings.world_storage.device = device
    with (
      patch.object(wp, "get_stream", return_value=SimpleNamespace(is_capturing=False)),
      patch.object(native_execution, "_validate_bindings"),
      patch.object(native_workspace, "step_workspace_layout", side_effect=AssertionError("Unexpected scratch planning")),
    ):
      for domain, name in (("world_storage", "nworld"), ("contact_storage", "naconmax"), ("ccd_storage", "naccdmax")):
        storage = getattr(bindings, domain)
        storage.capacity = 18
        try:
          with self.subTest(domain=domain), self.assertRaisesRegex(ValueError, f"Data.{name}"):
            make_step_workspace(None, data, arrays={}, bindings=bindings)
        finally:
          storage.capacity = 17

  def test_every_exported_prepared_entry_validates_or_rejects_before_work(self):
    """Keep the public prepared-stage surface complete and guard every callable before writes."""
    import mujoco_warp as mjw

    expected = {
      "step",
      "collision",
      "nxn_broadphase",
      "make_constraint",
      "deriv_smooth_vel",
      "forward",
      "fwd_actuation",
      "fwd_position",
      "fwd_velocity",
      "implicit",
      "island",
      "passive",
      "crb",
      "rne",
      "transmission",
      "solve",
    }
    exported = {
      name: function
      for name, function in vars(mjw).items()
      if inspect.isfunction(function)
      and "workspace" in inspect.signature(function).parameters
      and inspect.signature(function).parameters["workspace"].kind == inspect.Parameter.KEYWORD_ONLY
    }
    self.assertEqual(set(exported), expected)
    for name, program in exported.items():
      validate = Mock(side_effect=ValueError("invalid prepared binding"))
      arguments = [
        object()
        for parameter in inspect.signature(program).parameters.values()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
      ]
      with self.subTest(program=name), ExitStack() as stack:
        stack.enter_context(patch.object(native_execution, "validate_step_workspace", validate))
        for operation in ("launch", "launch_tiled", "empty", "empty_like", "zeros", "clone", "copy"):
          stack.enter_context(patch.object(wp, operation, side_effect=AssertionError("stage work before validation")))
        with self.assertRaisesRegex(ValueError, "invalid prepared binding"):
          program(*arguments, workspace=SimpleNamespace(bindings=None))
        validate.assert_called_once()

  def test_prepared_position_rejects_unprepared_factorization(self):
    """Reject the eager default factorization path before any position work is emitted."""
    workspace = SimpleNamespace(bindings=None)
    with (
      patch.object(wp, "launch", side_effect=AssertionError("work before rejection")),
      patch.object(native_execution, "validate_step_workspace"),
    ):
      with self.assertRaisesRegex(NotImplementedError, "factorize=False"):
        forward.fwd_position(object(), object(), workspace=workspace)

  def test_collision_provider_binds_native_scratch_once_and_preserves_external_signature(self):
    for prepared, native in ((False, False), (False, True), (True, True)):
      calls, snapshot = [], object()

      def external_collision(model, data, awake_prev=None):
        calls.append((model, data, awake_prev))

      def native_collision(model, data, awake_prev=None, *, workspace=None):
        calls.append((model, data, awake_prev, workspace))

      model = Mock(
        opt=SimpleNamespace(enableflags=types.EnableBit.SLEEP, disableflags=0, run_collision_detection=native),
        callback=SimpleNamespace(collision=None if native else external_collision),
        neq=0,
      )
      data = Mock()
      workspace = (
        SimpleNamespace(
          bindings=object(),
          arrays=dict(awake_prev=snapshot, efc_nnz=object(), island_parent=object(), moment_nnz=object()),
        )
        if prepared
        else None
      )
      with self.subTest(prepared=prepared, native=native), ExitStack() as stack:
        for module, name in (
          (native_execution, "validate_step_workspace"),
          (native_execution, "copy_step_rows"),
          (forward, "fwd_kinematics"),
          (smooth, "crb"),
          (smooth, "tendon_armature"),
          (sleep, "wake_collision"),
          (sleep, "update_sleep"),
          (constraint, "make_constraint"),
          (constraint, "_make_constraint"),
          (island, "island"),
          (island, "_island"),
          (smooth, "transmission"),
          (smooth, "_compute_transmission"),
        ):
          stack.enter_context(patch.object(module, name))
        stack.enter_context(patch.object(wp, "clone", return_value=snapshot))
        stack.enter_context(patch.object(forward.collision_driver, "collision", side_effect=native_collision))
        forward.fwd_position(model, data, factorize=False, workspace=workspace)
      suffix = (workspace,) if native else ()
      self.assertEqual(calls, [(model, data, None, *suffix), (model, data, snapshot, *suffix)])

  def test_standalone_stages_reuse_their_only_prepared_scratch_source(self):
    """Use the workspace's actuator and island scratch without hidden stage allocations."""
    parent, velocity = object(), object()
    workspace = SimpleNamespace(
      arrays={"island_parent": parent, "actuator_vel": velocity},
      bindings=object(),
    )
    with (
      patch.object(island, "direct_dsu") as launch,
      patch.object(wp, "empty", side_effect=AssertionError("scratch allocated")),
      patch.object(native_execution, "validate_step_workspace"),
    ):
      island.island(SimpleNamespace(ntree=2), SimpleNamespace(nworld=3), workspace=workspace)
      self.assertIs(launch.call_args.args[2], parent)
    model = Mock(opt=SimpleNamespace(disableflags=0, timestep=None), nactuator=2, has_fluid=False)
    data = Mock(nworld=3)
    with (
      patch.object(wp, "launch") as launch,
      patch.object(wp, "empty", side_effect=AssertionError("scratch allocated")),
      patch.object(native_execution, "validate_step_workspace"),
      patch.object(native_execution, "fill_step_rows"),
    ):
      derivative.deriv_smooth_vel(model, data, object(), workspace=workspace)
      self.assertIs(launch.call_args_list[0].kwargs["outputs"][0], velocity)
    for function, name, position in (
      (constraint.make_constraint, "efc_nnz", -1),
      (smooth.transmission, "moment_nnz", -1),
    ):
      workspace.arrays[name] = object()
      model = Mock(opt=SimpleNamespace(solver=types.SolverType.NEWTON, disableflags=types.DisableBit.CONSTRAINT), nacttrnbody=0)
      with (
        patch.object(native_execution, "validate_step_workspace"),
        patch.object(native_execution, "fill_step_rows") as fill,
        patch.object(wp, "launch", side_effect=RuntimeError("first stage launch")) as launch,
        patch.object(wp, "empty", side_effect=AssertionError("scratch allocated")),
        patch.object(wp, "zeros", side_effect=AssertionError("scratch allocated")),
        self.assertRaisesRegex(RuntimeError, "first stage launch"),
      ):
        function(model, data, workspace=workspace)
      self.assertIs(launch.call_args.kwargs["inputs"][position], workspace.arrays[name])
      if name == "moment_nnz":
        fill.assert_called_once_with(workspace.bindings, workspace.arrays[name], 0, "world")

  def test_shared_computations_have_explicit_operands_without_storage_or_validation(self):
    """Keep computation independent of its eager or prepared allocation boundary."""
    for operation in (
      constraint._make_constraint,
      island._island,
      island.direct_dsu,
      smooth._compute_transmission,
      derivative._deriv_smooth_vel,
    ):
      with self.subTest(operation=operation.__name__):
        tree = ast.parse(inspect.getsource(operation))
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        calls = {ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
        self.assertFalse({"workspace", "validated", "skip_validation"} & names)
        self.assertFalse(
          {"wp.empty", "wp.zeros", "wp.empty_like", "wp.clone", "step_execution.validate_step_workspace"} & calls
        )

  def test_prepared_composition_passes_scratch_without_reentering_public_stages(self):
    """Validate each composing entry, then call computations with the exact borrowed arrays."""
    model = Mock(
      opt=SimpleNamespace(
        enableflags=types.EnableBit.SLEEP,
        disableflags=0,
        run_collision_detection=False,
        integrator=types.IntegratorType.IMPLICITFAST,
      ),
      callback=SimpleNamespace(collision=None),
      neq=0,
      body_freeadr=SimpleNamespace(size=0),
    )
    data = Mock()
    workspace = SimpleNamespace(
      bindings=object(),
      arrays={
        name: object()
        for name in (
          "efc_nnz",
          "island_parent",
          "moment_nnz",
          "actuator_vel",
          "qDeriv",
          "qLD",
          "qLDiagInv",
          "qacc",
          "island_can_sleep",
        )
      },
    )
    with ExitStack() as stack:
      validate = stack.enter_context(patch.object(native_execution, "validate_step_workspace"))
      for module, name in (
        (constraint, "make_constraint"),
        (island, "island"),
        (smooth, "transmission"),
        (derivative, "deriv_smooth_vel"),
      ):
        stack.enter_context(patch.object(module, name, side_effect=AssertionError("Repeated public stage boundary")))
      for module, name in (
        (forward, "fwd_kinematics"),
        (smooth, "crb"),
        (smooth, "tendon_armature"),
        (forward.sleep, "wake_collision"),
        (forward.sleep, "update_sleep"),
        (smooth, "factor_solve_i"),
        (forward, "_launch_implicit_free_body_solve"),
        (forward, "_advance"),
      ):
        stack.enter_context(patch.object(module, name))
      operations = [
        stack.enter_context(patch.object(module, name))
        for module, name in (
          (constraint, "_make_constraint"),
          (island, "_island"),
          (smooth, "_compute_transmission"),
          (derivative, "_deriv_smooth_vel"),
        )
      ]
      forward.fwd_position(model, data, factorize=False, workspace=workspace)
      validate.assert_called_once_with(workspace, model, data)
      operations[0].assert_called_once_with(model, data, workspace.arrays["efc_nnz"], bindings=workspace.bindings)
      operations[1].assert_called_once_with(model, data, workspace.arrays["island_parent"])
      operations[2].assert_called_once_with(
        model, data, workspace.arrays["moment_nnz"], body_ncon=None, bindings=workspace.bindings
      )
      validate.reset_mock()
      forward.implicit(model, data, workspace=workspace)
      validate.assert_called_once_with(workspace, model, data)
      operations[3].assert_called_once_with(
        model, data, workspace.arrays["qDeriv"], workspace.arrays["actuator_vel"], bindings=workspace.bindings
      )

  def test_transmission_initializes_supplied_counters_on_every_execution(self):
    """Retain first-write semantics when allocation moves outside the numerical body."""
    model, data = Mock(nacttrnbody=2), Mock(nworld=3)
    moment_nnz = wp.empty(3, dtype=int, device="cpu")
    body_ncon = wp.empty((3, 2), dtype=int, device="cpu")
    with patch.object(wp, "launch"):
      for poison in (17, -91):
        moment_nnz.fill_(poison)
        body_ncon.fill_(poison)
        smooth._compute_transmission(model, data, moment_nnz, body_ncon=body_ncon, bindings=None)
        np.testing.assert_array_equal(moment_nnz.numpy(), 0)
        np.testing.assert_array_equal(body_ncon.numpy(), 0)

  def test_prepared_stages_reject_alternate_scratch_before_writing_or_launching(self):
    """One prepared scratch owner cannot be bypassed through an optional eager argument."""
    for function, arguments, parameter, field in (
      (constraint.make_constraint, (object(), object()), "efc_nnz", "efc_nnz"),
      (smooth.transmission, (object(), object()), "moment_nnz", "moment_nnz"),
      (island.island, (object(), object()), "parent", "island_parent"),
      (derivative.deriv_smooth_vel, (object(), object(), object()), "actuator_vel", "actuator_vel"),
    ):
      workspace = SimpleNamespace(bindings=object(), arrays={field: object()})
      self.assertNotIn(parameter, inspect.signature(function).parameters)
      with (
        self.subTest(function=function.__name__),
        patch.object(native_execution, "validate_step_workspace"),
        patch.object(native_execution, "fill_step_rows") as fill,
        patch.object(wp, "launch", side_effect=AssertionError("unadmitted scratch touched")),
        self.assertRaisesRegex(TypeError, "unexpected keyword"),
      ):
        function(*arguments, **{parameter: object()}, workspace=workspace)
      fill.assert_not_called()

  def test_native_program_rejects_dense_inertia_override_before_publication(self):
    """The computation keeps its eager API; prepared admission owns supported kernel paths."""
    model, data = Mock(nv=2), Mock(nworld=17)
    for rank in (2, 3):
      bindings = _step_bindings()
      with self.subTest(rank=rank), patch.object(wp, "launch") as launch, wp.ScopedDevice("cpu"):
        support.mul_m(model, data, object(), object(), M=SimpleNamespace(ndim=rank))
      call = launch.call_args
      self.assertEqual(call.kwargs["inputs"][-1].size, 0)
      self.assertEqual(call.kwargs["dim"], (17, 2))
      record = wp.CapturedLaunch(call.args[0], call.kwargs["dim"], 1, 120, (), (), 128, 0, False)
      with (
        _binding_capture(bindings) as graph,
        patch.object(wp, "capture_get_launches", return_value=(record,)),
        patch.object(step_program.field_ops, "validate_captured_operation", return_value=False),
        patch.object(step_program.graph_ops, "adopt_launch", return_value=object()) as adopt,
      ):
        if rank == 2:
          step_program.bind_step_program(bindings, graph, (record,))
          adopt.assert_called_once()
          self.assertIs(adopt.call_args.kwargs["extent_source"], bindings.world_storage.protected_count)
        else:
          with self.assertRaisesRegex(ValueError, "no native count declaration"):
            step_program.bind_step_program(bindings, graph, (record,))
          adopt.assert_not_called()
          self.assertTrue(bindings.recording_failed)
          self.assertTrue(graph._preparation_failed)

  def test_prepared_execution_binding_cannot_change_mode_count_or_bindings(self):
    """Keep the sole count in the owner record and reject mode or descriptor replacement."""
    bindings = _step_bindings()
    count = bindings.world_storage.protected_count
    specs = _collision_scratch_specs()
    arrays = {spec.name: wp.empty(spec.shape, dtype=spec.dtype, device="cpu") for spec in specs}
    data = _workspace_data(count)
    with _cpu_workspace_preparation(specs), patch.object(native_workspace.field_ops, "lookup", return_value=0):
      dynamic = make_step_workspace(Metadata(), data, arrays=arrays, bindings=bindings)
      fixed = make_step_workspace(Metadata(), data, arrays=arrays)
    self.assertFalse(hasattr(dynamic, "world_live_count"))
    for replacement in (None, _step_bindings()):
      dynamic.bindings = replacement
      with self.assertRaisesRegex(ValueError, "execution binding changed"):
        native_execution.validate_step_workspace(dynamic, dynamic.model, data)
    dynamic.bindings = bindings
    count.fill_(1)
    native_execution.validate_step_workspace(dynamic, dynamic.model, data)
    for name, changed in (("shape", (0,)), ("strides", (8,)), ("dtype", wp.int64)):
      original = getattr(count, name)
      try:
        setattr(count, name, changed)
        with self.assertRaisesRegex(ValueError, "storage or count descriptors changed"):
          native_execution.validate_step_workspace(dynamic, dynamic.model, data)
      finally:
        setattr(count, name, original)
    for owner, name, value in (
      (bindings, "world_storage", _step_bindings().world_storage),
      (bindings.world_storage, "protected_count", wp.zeros(1, dtype=wp.int32, device="cpu")),
      (bindings.contact_storage, "ready_count", wp.zeros(1, dtype=wp.int32, device="cpu")),
      (bindings.ccd_storage, "capacity", 18),
      (bindings, "bindings", []),
    ):
      original = getattr(owner, name)
      try:
        setattr(owner, name, value)
        with self.assertRaisesRegex(ValueError, "storage or count descriptors changed"):
          native_execution.validate_step_workspace(dynamic, dynamic.model, data)
      finally:
        setattr(owner, name, original)
    native_execution.validate_step_workspace(dynamic, dynamic.model, data)
    fixed.bindings = bindings
    with self.assertRaisesRegex(ValueError, "execution binding changed"):
      native_execution.validate_step_workspace(fixed, fixed.model, data)

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
    before = native_execution._layout(model), native_execution._layout(template)
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
    self.assertEqual(before, (native_execution._layout(model), native_execution._layout(template)))

  def test_caller_scratch_binding_owns_no_allocation(self):
    specs = _collision_scratch_specs()
    arrays = {spec.name: wp.empty(spec.shape, dtype=spec.dtype, device="cpu") for spec in specs}
    data = _workspace_data(arrays["collision_worldid"])
    with (
      _cpu_workspace_preparation(specs),
      patch.object(wp, "empty", side_effect=AssertionError("Unexpected allocation")),
      patch.object(native_workspace.field_ops, "contiguous", side_effect=AssertionError("Borrowed scratch planned allocation")),
      patch.object(native_workspace.field_ops, "pack", side_effect=AssertionError("Borrowed scratch packed a buffer")),
      patch.object(native_workspace.field_ops, "span_bytes", side_effect=AssertionError("Borrowed scratch planned byte spans")),
    ):
      workspace = make_step_workspace(Metadata(), data, arrays=arrays)
    self.assertIsNone(workspace.storage)
    report = native_workspace.step_workspace_memory_report(workspace)
    self.assertEqual(report["allocation_owner"], "caller")
    for name in ("solver_model", "solver_data", "solver_context", "collision"):
      self.assertFalse(hasattr(workspace, name), "Derived stage views belong to the engine, not caller configuration")
    self.assertIsNone(report["physical_scratch_bytes"])
    self.assertEqual(
      report["payload_bytes"], sum(array.size * wp.types.type_size_in_bytes(array.dtype) for array in arrays.values())
    )
    domains = {spec.name: spec.capacity_domain for spec in specs}
    for field in report["fields"]:
      self.assertEqual(field["capacity_domain"], domains[field["name"]])
      self.assertIsNone(field["allocation_offset_bytes"])
      self.assertFalse({"scope", "offset", "bytes"} & field.keys())
    for name, array in arrays.items():
      self.assertIs(workspace.arrays[name], array)
    arrays["collision_worldid"].shape = (1,)
    self.assertEqual(native_workspace.step_workspace_memory_report(workspace), report)
    with self.assertRaisesRegex(ValueError, "scratch descriptors"):
      native_execution.validate_step_workspace(workspace, workspace.model, data)

  def test_snapshot_aliases_share_one_traversal_but_later_mutations_remain_visible(self):
    array = wp.empty(2, dtype=wp.float32, device="cpu")
    metadata = Metadata(pair_counts=(array,))
    first = native_execution._layout((metadata, metadata, array))
    self.assertIs(first[1][0], native_execution._LAYOUT_REFERENCE)
    self.assertIs(first[2][0], native_execution._LAYOUT_REFERENCE)
    self.assertNotEqual(first[1], first[2])
    self.assertEqual(first, native_execution._layout((metadata, metadata, array)))
    split_alias = dataclasses.replace(metadata)
    self.assertNotEqual(first, native_execution._layout((metadata, split_alias, array)))
    metadata.option += 1
    self.assertNotEqual(first, native_execution._layout((metadata, metadata, array)))
    metadata.option -= 1
    array.shape = (1,)
    self.assertNotEqual(first, native_execution._layout((metadata, metadata, array)))
    array.shape = (2,)
    self.assertEqual(first, native_execution._layout((metadata, metadata, array)))

    # A short-lived traversal retains temporary roots, preventing recycled IDs.
    temporary = Metadata()
    reference = weakref.ref(temporary)
    memo = {}
    native_execution._layout(temporary, memo)
    del temporary
    self.assertIsNotNone(reference())
    del memo
    self.assertIsNone(reference())

  def test_derived_scratch_record_rebinding_rejected_before_capture(self):
    """The shared arrays and the stage records must describe the same retained allocations."""
    specs = _collision_scratch_specs()
    arrays = {spec.name: wp.empty(spec.shape, dtype=spec.dtype, device="cpu") for spec in specs}
    data = _workspace_data(arrays["collision_worldid"])
    with _cpu_workspace_preparation(specs):
      workspace = make_step_workspace(Metadata(), data, arrays=arrays)
    replacement = wp.empty_like(arrays["nccd"])
    for name, field in (("convex", "nccd"), ("_collision", "collision_worldid")):
      original = getattr(workspace, name)
      try:
        setattr(workspace, name, dataclasses.replace(original, **{field: replacement}))
        with self.subTest(name=name), self.assertRaisesRegex(ValueError, "scratch descriptors"):
          native_execution.validate_step_workspace(workspace, workspace.model, data)
      finally:
        setattr(workspace, name, original)
    workspace._solver_context = SimpleNamespace()
    with self.assertRaisesRegex(ValueError, "scratch descriptors"):
      native_execution.validate_step_workspace(workspace, workspace.model, data)

  def test_caller_scratch_invalid_descriptors_reject_before_allocating(self):
    owner = wp.empty(128, dtype=wp.uint8, device="cpu")
    spec = WorkspaceFieldSpec("scratch", (2, 3), wp.float32, "world")
    data = _workspace_data(owner)
    valid = wp.array(ptr=owner.ptr, shape=(2, 3), strides=(32, 4), dtype=wp.float32, device="cpu")
    cases = (
      {},
      {"scratch": valid, "extra": valid},
      {"scratch": wp.empty((2, 3), dtype=wp.int32, device="cpu")},
      {"scratch": wp.empty((2, 2), dtype=wp.float32, device="cpu")},
      {"scratch": wp.array(ptr=owner.ptr, shape=(2, 3), strides=(32, 8), dtype=wp.float32, device="cpu")},
      {"scratch": wp.array(ptr=owner.ptr, shape=(2, 3), strides=(4, 4), dtype=wp.float32, device="cpu")},
    )
    with _cpu_workspace_preparation((spec,)), patch.object(wp, "empty", side_effect=AssertionError("Unexpected allocation")):
      for arrays in cases:
        with self.assertRaises(ValueError):
          make_step_workspace(Metadata(), data, arrays=arrays)

  def test_prepared_scalar_mutation_rejected_before_capture(self):
    for owner, field in (("data", "nworld"), ("data", "njmax"), ("data", "naccdmax"), ("model", "nv"), ("model", "option")):
      model, data = Metadata(), Metadata()
      workspace = dataclasses.replace(
        self.metadata_workspace,
        model=model,
        data=data,
        _data_layout=native_execution._layout(data),
        _model_layout=native_execution._layout(model),
      )
      value = getattr(workspace, owner)
      setattr(value, field, getattr(value, field) + 1)
      with self.subTest(owner=owner, field=field), self.assertRaisesRegex(ValueError, "scalar metadata"):
        native_execution.validate_step_workspace(workspace, model, data)

  def test_foreign_model_or_data_rejected(self):
    workspace = self.metadata_workspace
    for model, data in ((Metadata(), workspace.data), (workspace.model, Metadata())):
      with self.assertRaisesRegex(ValueError, "original model"):
        native_execution.validate_step_workspace(workspace, model, data)

  def test_nested_list_array_rebinding_is_part_of_immutable_metadata(self):
    """Inspect supported list containers recursively instead of trusting their Python identity."""
    for owner in ("model", "data"):
      model, data = Metadata(), Metadata()
      nested = [wp.zeros(1, dtype=float, device="cpu")]
      setattr(model if owner == "model" else data, "pair_counts", [nested])
      workspace = dataclasses.replace(
        self.metadata_workspace,
        model=model,
        data=data,
        _data_layout=native_execution._layout(data),
        _model_layout=native_execution._layout(model),
      )
      nested[0] = wp.ones(1, dtype=float, device="cpu")
      with self.subTest(owner=owner), self.assertRaisesRegex(ValueError, "scalar metadata"):
        native_execution.validate_step_workspace(workspace, model, data)

  def test_prepared_topology_tuple_or_callback_change_rejected(self):
    for field, replacement in (("pair_counts", (2, 2)), ("callback", lambda *_: None)):
      workspace = dataclasses.replace(self.metadata_workspace, model=Metadata())
      setattr(workspace.model, field, replacement)
      with self.assertRaisesRegex(ValueError, "scalar metadata"):
        native_execution.validate_step_workspace(workspace, workspace.model, workspace.data)

  def test_blocked_scratch_alignment_rejects_base_and_world_stride(self):
    """Reject misaligned blocked matrices without restricting dense scalar counters."""
    specs = (
      *_collision_scratch_specs(),
      WorkspaceFieldSpec("solver.h", (2, 64, 64), wp.float32, "world"),
      WorkspaceFieldSpec("solver.hfactor", (2, 64, 64), wp.float32, "world"),
      WorkspaceFieldSpec("moment_nnz", (2,), wp.int32, "world"),
    )
    owner = wp.empty(65536, dtype=wp.uint8, device="cpu")
    baseline = {spec.name: wp.empty(spec.shape, dtype=spec.dtype, device="cpu") for spec in specs}
    data, model = _workspace_data(baseline["moment_nnz"]), Metadata(nv=64)
    for name in ("solver.h", "solver.hfactor"):
      for base, stride in ((4, 16384), (0, 16388)):
        arrays = dict(baseline)
        arrays[name] = wp.array(
          ptr=owner.ptr + base, shape=(2, 64, 64), strides=(stride, 256, 4), dtype=wp.float32, device="cpu"
        )
        with self.subTest(name=name, base=base, stride=stride), _cpu_workspace_preparation(specs):
          with self.assertRaisesRegex(ValueError, "16-byte"):
            make_step_workspace(model, data, arrays=arrays)
    with _cpu_workspace_preparation(specs), patch.object(wp, "empty", side_effect=AssertionError("Unexpected allocation")):
      workspace = make_step_workspace(model, data, arrays=baseline)
    self.assertEqual(workspace.arrays["moment_nnz"].strides, (4,))
    # The nonblocked path does not request aligned=True; an absent hfactor is valid.
    small = (
      *_collision_scratch_specs(),
      WorkspaceFieldSpec("solver.h", (2, 2, 2), wp.float32, "world"),
      WorkspaceFieldSpec("solver.hfactor", (2, 0, 0), wp.float32, "world"),
      specs[-1],
    )
    arrays = {spec.name: wp.empty(spec.shape, dtype=spec.dtype, device="cpu") for spec in small}
    arrays["solver.h"] = wp.array(ptr=owner.ptr + 4, shape=(2, 2, 2), strides=(20, 8, 4), dtype=wp.float32, device="cpu")
    with _cpu_workspace_preparation(small):
      make_step_workspace(Metadata(nv=2), data, arrays=arrays)

  def test_compact_data_matrix_alignment_rejects_each_address_axis(self):
    """Reject misaligned compact matrix bases/strides before scratch binding."""
    owner = wp.empty(65536, dtype=wp.uint8, device="cpu")
    valid = wp.array(ptr=owner.ptr, shape=(2, 64, 64), strides=(16384, 256, 4), dtype=wp.float32, device="cpu")
    data = _workspace_data(owner)
    data.cM = data.cqLD = valid
    with _cpu_workspace_preparation(()), patch.object(wp, "empty", side_effect=AssertionError("Unexpected allocation")):
      for name in ("cM", "cqLD"):
        for base, world_stride, row_stride in ((4, 16384, 256), (0, 16388, 256), (0, 16640, 260)):
          setattr(
            data,
            name,
            wp.array(
              ptr=owner.ptr + base, shape=(2, 64, 64), strides=(world_stride, row_stride, 4), dtype=wp.float32, device="cpu"
            ),
          )
          with self.subTest(name=name, base=base, stride=world_stride), self.assertRaisesRegex(ValueError, "16-byte"):
            make_step_workspace(None, data)
          setattr(data, name, valid)
    solver.validate_blocked_matrix(valid)
    solver.validate_blocked_matrix(wp.empty((2, 0, 0), dtype=wp.float32, device="cpu"))

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
