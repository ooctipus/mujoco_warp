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
from mujoco_warp._src import step_program
from mujoco_warp._src import support
from mujoco_warp._src import types
from mujoco_warp._src import warp_util
from mujoco_warp._src import workspace as native_workspace
from mujoco_warp._src.step_program import StepBindings
from mujoco_warp._src.step_program import resolve_step_counts
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


def _minimal_scratch_specs():
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
    WorkspaceFieldSpec("awake_changed", (1,), wp.int32, "global_counter"),
    WorkspaceFieldSpec("nsolving", (1,), wp.int32, "global_counter"),
    *(WorkspaceFieldSpec(name, (17,), wp.int32, "world") for name in ("efc_nnz", "moment_nnz")),
    *(
      WorkspaceFieldSpec(name, (17, 1), wp.int32, "world")
      for name in ("awake_prev", "island_parent", "island_can_sleep", "efc_tree")
    ),
    *(
      WorkspaceFieldSpec(name, (17, 1), wp.float32, "world") for name in ("qDeriv", "qLD", "qLDiagInv", "qacc", "actuator_vel")
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


def _program_workspace(bindings):
  data = SimpleNamespace(nacon=wp.zeros(1, dtype=int, device="cpu"), ncollision=wp.zeros(1, dtype=int, device="cpu"))
  return SimpleNamespace(bindings=bindings, model=Metadata(), data=data, arrays={}, _specs=())


@contextmanager
def _binding_capture(bindings):
  bindings.updates = _update_table(bindings)

  class CapturedGraph:
    pass

  graph = CapturedGraph()
  graph.device = bindings.world_storage.device

  with (
    patch.object(step_program.graph_ops, "current_capture", return_value=graph),
    patch.object(wp, "get_stream", return_value=None),
  ):
    yield graph


class NativeBindingsTest(unittest.TestCase):
  def test_physics_launches_stay_plain_and_prepared_signatures_have_resource_purpose(self):
    """Gate the native execution boundary independently of numerical output parity."""
    for path in Path(step_program.__file__).parent.glob("*.py"):
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
    self.assertFalse(hasattr(step_program, "bind_step_launch"))
    self.assertFalse(hasattr(step_program, "_record_launch"))
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
    self.assertIs(typing.get_type_hints(StepBindings)["world_storage"], FieldStorage)
    self.assertEqual(typing.get_type_hints(StepBindings)["updates"], GraphUpdateTable | None)
    self.assertFalse(any(name in inspect.getsource(step_program) for name in ("import newton", "recorder=")))

  def test_numerical_modules_have_no_execution_or_workspace_coupling(self):
    for module in (forward.collision_driver, constraint, derivative, forward, island, passive, sleep, smooth, solver):
      with self.subTest(module=module.__name__):
        tree = ast.parse(inspect.getsource(module))
        for node in ast.walk(tree):
          if isinstance(node, ast.Name):
            self.assertNotIn(node.id, ("workspace", "bindings", "step_execution", "step_program", "gpu_components"))
          elif isinstance(node, ast.ImportFrom):
            self.assertFalse((node.module or "").startswith("gpu_components"))
            self.assertFalse(any(alias.name in ("workspace", "step_execution", "step_program") for alias in node.names))
    for name in ("fill_step_rows", "copy_step_rows", "_begin_recording"):
      self.assertFalse(hasattr(step_program, name))
    for module, name in (
      (constraint, "_make_constraint"),
      (derivative, "_deriv_smooth_vel"),
      (island, "_island"),
      (smooth, "_compute_transmission"),
    ):
      self.assertFalse(hasattr(module, name), "Allocation-only wrappers must not survive as compatibility layers")

  def test_execution_dependencies_and_exports_have_one_canonical_owner(self):
    self.assertEqual(StepBindings.__module__, step_program.__name__)
    self.assertFalse(Path(step_program.__file__).with_name("step_execution.py").exists())
    tree = ast.parse(inspect.getsource(step_program))
    for node in ast.walk(tree):
      if isinstance(node, ast.Import):
        imports = [alias.name for alias in node.names]
      elif isinstance(node, ast.ImportFrom):
        imports = [node.module or "", *(f"{node.module}.{alias.name}" for alias in node.names)]
      else:
        continue
      self.assertNotIn("mujoco_warp._src.workspace", imports, "Preparation depends on program declarations, never the reverse")
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
    self.assertNotIn("mujoco_warp._src.step_program", imports)

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
    with patch.object(step_program.graph_ops, "register_last_kernel_node", side_effect=AssertionError("node claimed")):
      for domain, axis, parameters in cases:
        with self.subTest(domain=domain, axis=axis, parameters=parameters), self.assertRaises(ValueError):
          resolve_step_counts(bindings, kernel, domain, axis, parameters)
      kernel.adj.args[1].type = wp.int64
      with self.assertRaisesRegex(ValueError, "int32"):
        resolve_step_counts(bindings, kernel, "world", 0, {"world_live_count": "world"})
    self.assertFalse(bindings.recording_failed)
    self.assertEqual(bindings.bindings, [])
    self.assertIsNone(bindings._recording_binding)

  def test_recording_rejects_replaced_sources_updater_or_bindings_but_allows_count_values(self):
    bindings, kernel = _step_bindings(), _native_kernel()
    with _binding_capture(bindings) as graph:
      step_program._retain_bindings(bindings, graph)
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
      step_program._retain_bindings(bindings, graph)
      bindings.world_storage = object()
      with self.assertRaisesRegex(TypeError, "borrow FieldStorage"):
        resolve_step_counts(bindings, _native_kernel(), "world", 0, {})
      self.assertFalse(getattr(graph, "_preparation_failed", False))


class StepProgramTest(unittest.TestCase):
  def test_native_domains_resolve_independent_sources_in_one_batch(self):
    bindings = _step_bindings()
    bindings.contact_storage.capacity, bindings.ccd_storage.capacity = 19, 23
    kernels = (
      smooth._kinematics_branch,
      sleep._wake_collision_kernel,
      collision_convex.ccd_kernel_builder(6, 6, 35, 16, False, 0, 32, 0),
      forward._next_time_builder(0),
    )
    records = tuple(
      wp.CapturedLaunch(kernel, dim, index + 1, 120, (), (), 128, 0, False)
      for index, (kernel, dim) in enumerate(zip(kernels, ((17, 4), (19,), (8192,), (17,))))
    )
    warp_util.clear_kernel_cache()
    with (
      _binding_capture(bindings) as graph,
      patch.object(step_program, "validate_step_workspace"),
      patch.object(wp, "capture_get_memory_operations", return_value=()),
      patch.object(step_program.graph_ops, "adopt_launches", return_value=(object(),) * 4) as adopt,
    ):
      workspace = _program_workspace(bindings)
      step_program.bind_step_program(workspace, graph, records)
      adopt.assert_called_once()
      self.assertEqual(adopt.call_args.args, (bindings.updates, graph, records))
      extents = adopt.call_args.kwargs["extents"]
      self.assertEqual([(index, axis) for index, axis, _ in extents], [(0, 0), (1, 0), (3, 0)])
      self.assertIs(extents[0][2], bindings.world_storage.protected_count)
      self.assertIs(extents[1][2], bindings.contact_storage.ready_count)
      for index, parameter in adopt.call_args.kwargs["parameters"]:
        self.assertIn(index, (2, 3))
        label = records[index].kernel.adj.args[parameter.argument_index - 1].label
        expected = {
          "nworld_in": bindings.world_storage.protected_count,
          "naconmax_in": bindings.contact_storage.ready_count,
          "naccdmax_in": bindings.ccd_storage.ready_count,
        }
        self.assertIs(parameter.source, expected[label])
      self.assertEqual(len(bindings.bindings), 4)
      self.assertEqual(adopt.call_args.kwargs["fixed"], [])

  def test_native_declaration_or_adoption_failures_prevent_publication(self):
    kernel = smooth._kinematics_branch
    record = wp.CapturedLaunch(kernel, (17, 4), 1, 120, (), (), 128, 0, False)
    for problem in ("unknown", "capacity", "mutable_sequence", "adoption"):
      bindings = _step_bindings()
      launches = (record,)
      if problem == "unknown":
        launches = (dataclasses.replace(record, kernel=Mock(key=kernel.key)),)
      elif problem == "capacity":
        launches = (dataclasses.replace(record, dim=(16, 4)),)
      elif problem == "mutable_sequence":
        launches = [record]
      with (
        self.subTest(problem=problem),
        _binding_capture(bindings) as graph,
        patch.object(step_program, "validate_step_workspace"),
        patch.object(wp, "capture_get_memory_operations", return_value=()),
        patch.object(step_program.graph_ops, "adopt_launches", side_effect=ArithmeticError("failed adoption")) as adopt,
      ):
        with self.assertRaises((ValueError, TypeError, ArithmeticError)):
          step_program.bind_step_program(_program_workspace(bindings), graph, launches)
        self.assertTrue(bindings.recording_failed)
        self.assertTrue(graph._preparation_failed)
        self.assertEqual(bindings.bindings, [])
        self.assertEqual(adopt.call_count, int(problem == "adoption"))

  def test_memory_records_require_exact_domains_and_explicit_global_operands(self):
    record = wp.CapturedLaunch(Mock(key="memory_operation"), (17,), 1, 120, (), (), 128, 0, False)
    operation = SimpleNamespace(launch=record)
    for mode in ("dynamic", "fixed", "invalid"):
      bindings = _step_bindings()
      workspace = _program_workspace(bindings)
      counter = wp.zeros(1, dtype=int, device="cpu")
      workspace.arrays = {"nsolving": counter}
      workspace._specs = (WorkspaceFieldSpec("nsolving", (1,), wp.int32, "global_counter"),)
      with (
        self.subTest(mode=mode),
        _binding_capture(bindings) as graph,
        patch.object(step_program, "validate_step_workspace"),
        patch.object(wp, "capture_get_memory_operations", return_value=(operation,)),
        patch.object(
          step_program.field_ops,
          "memory_operation_extents",
          return_value=((0, bindings.world_storage.protected_count),) if mode == "dynamic" else ((None, None),),
          side_effect=ValueError("foreign operand") if mode == "invalid" else None,
        ) as extent,
        patch.object(step_program.graph_ops, "adopt_launches", return_value=(object(),)) as adopt,
      ):
        if mode == "invalid":
          with self.assertRaisesRegex(ValueError, "foreign operand"):
            step_program.bind_step_program(workspace, graph, (record,))
          adopt.assert_not_called()
          self.assertTrue(graph._preparation_failed)
        else:
          step_program.bind_step_program(workspace, graph, (record,))
          self.assertEqual(adopt.call_args.kwargs["fixed"], [0] if mode == "fixed" else [])
        args = extent.call_args
        self.assertIs(args.args[0], graph)
        self.assertEqual(args.args[1], (operation,))
        self.assertEqual(
          [id(owner) for owner, _ in args.args[2]],
          [id(bindings.world_storage), id(bindings.contact_storage), id(bindings.ccd_storage)],
        )
        self.assertEqual(
          {id(array) for array in args.kwargs["fixed_arrays"]},
          {id(workspace.data.nacon), id(workspace.data.ncollision), id(bindings.world_storage.protected_count), id(counter)},
        )

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
      scratch=None,
      _specs=(),
      _allocation_offsets_bytes=None,
      _execution_binding=None,
      _binding_layout=None,
      _data_layout=step_program._layout(data),
      _model_layout=step_program._layout(model),
      _scratch_layout=step_program._layout(((), None)),
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

  def test_disabled_derivative_preserves_ordinary_copy_semantics(self):
    model = SimpleNamespace(
      has_fluid=False,
      M_fullm_i=None,
      M_fullm_j=None,
      opt=SimpleNamespace(disableflags=types.DisableBit.ACTUATION | types.DisableBit.DAMPER),
    )
    data = SimpleNamespace(M=wp.array(np.arange(12, dtype=np.float32).reshape(4, 3), device="cpu"))
    out = wp.full((4, 3), -7.0, device="cpu")
    with patch.object(wp, "launch", side_effect=AssertionError("Unexpected derivative kernel")):
      derivative.deriv_smooth_vel(model, data, out)
    np.testing.assert_array_equal(out.numpy(), data.M.numpy())

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
    )
    for operation, nactuator, disableflags, names in cases:
      with self.subTest(operation=operation.__name__, nactuator=nactuator):
        arrays = {name: wp.full((4, 3), 7.0, device="cpu") for name in names}
        model = SimpleNamespace(nactuator=nactuator, opt=SimpleNamespace(disableflags=disableflags))
        with patch.object(wp, "launch", side_effect=AssertionError("Unexpected stage work")):
          operation(model, SimpleNamespace(**arrays))
        for array in arrays.values():
          np.testing.assert_array_equal(array.numpy(), 0)

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
    count_source = "world_count"
    self.assertTrue(
      any(
        isinstance(node, ast.Call)
        and ast.unparse(node.func) == "wp.copy"
        and [ast.unparse(argument) for argument in node.args] == ["nsolving", count_source]
        for node in ast.walk(tree)
      ),
      "Prepared solver initializes its condition from the live count before entering the captured loop",
    )

  def test_solver_initializes_condition_each_call_without_mutating_world_count(self):
    condition = wp.empty(1, dtype=int, device="cpu")
    world_count = wp.zeros(1, dtype=int, device="cpu")
    model = Mock(opt=SimpleNamespace(disableflags=0, solver=types.SolverType.NEWTON, iterations=1, graph_conditional=True))
    with (
      patch.object(wp, "launch"),
      patch.object(solver, "init_context"),
      patch.object(solver, "_use_incremental", return_value=False),
    ):
      for live in (0, 3, 7):
        for source in (None, world_count):
          world_count.fill_(live)
          condition.fill_(-91)
          with patch.object(wp, "capture_while") as loop:
            solver._solve(model, Mock(nworld=7), Mock(), nsolving=condition, world_count=source)
          np.testing.assert_array_equal(condition.numpy(), [7 if source is None else live])
          np.testing.assert_array_equal(world_count.numpy(), [live])
          self.assertIs(loop.call_args.args[0], condition)

  def test_global_scratch_cannot_alias_borrowed_population_or_capacity_counts(self):
    specs = _minimal_scratch_specs()
    for owner, name in (
      ("world_storage", "protected_count"),
      ("contact_storage", "ready_count"),
      ("ccd_storage", "ready_count"),
    ):
      for scratch_name in ("nccd", "awake_changed", "nsolving"):
        bindings = _step_bindings()
        arrays = {spec.name: wp.empty(spec.shape, dtype=spec.dtype, device="cpu") for spec in specs}
        count = wp.array(ptr=arrays[scratch_name].ptr, shape=(1,), dtype=wp.int32, device="cpu")
        setattr(getattr(bindings, owner), name, count)
        data = _workspace_data(count)
        with (
          self.subTest(count=owner, scratch=scratch_name),
          _cpu_workspace_preparation(specs),
          patch.object(native_workspace.field_ops, "lookup", return_value=0),
        ):
          with self.assertRaisesRegex(ValueError, "overlap"):
            make_step_workspace(Metadata(), data, arrays=arrays, bindings=bindings)

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
      patch.object(step_program, "_validate_bindings"),
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

  def test_borrowed_position_scratch_rejects_unprepared_factorization(self):
    with patch.object(wp, "launch", side_effect=AssertionError("work before rejection")):
      with self.assertRaisesRegex(NotImplementedError, "factorize=False"):
        forward.fwd_position(object(), object(), scratch=object())

  def test_collision_provider_binds_native_scratch_once_and_preserves_external_signature(self):
    for prepared, native in ((False, False), (False, True), (True, True)):
      calls, snapshot = [], object()

      def external_collision(model, data, awake_prev=None):
        calls.append((model, data, awake_prev))

      def native_collision(model, data, awake_prev=None, *, scratch=None):
        calls.append((model, data, awake_prev, scratch))

      model = Mock(
        opt=SimpleNamespace(enableflags=types.EnableBit.SLEEP, disableflags=0, run_collision_detection=native),
        callback=SimpleNamespace(collision=None if native else external_collision),
        neq=0,
      )
      data = Mock()
      scratch = forward._PositionScratch(object(), snapshot, object(), object(), object()) if prepared else None
      with self.subTest(prepared=prepared, native=native), ExitStack() as stack:
        for module, name in (
          (wp, "copy"),
          (forward, "fwd_kinematics"),
          (smooth, "crb"),
          (smooth, "tendon_armature"),
          (sleep, "wake_collision"),
          (sleep, "update_sleep"),
          (constraint, "make_constraint"),
          (island, "island"),
          (smooth, "transmission"),
        ):
          stack.enter_context(patch.object(module, name))
        stack.enter_context(patch.object(wp, "clone", return_value=snapshot))
        stack.enter_context(patch.object(forward.collision_driver, "collision", side_effect=native_collision))
        forward.fwd_position(model, data, factorize=False, scratch=scratch)
      suffix = ((scratch.collision if prepared else None),) if native else ()
      self.assertEqual(calls, [(model, data, None, *suffix), (model, data, snapshot, *suffix)])

  def test_standalone_stages_borrow_exact_scratch_without_allocation(self):
    parent, velocity, nonzeros = object(), object(), object()
    with patch.object(wp, "empty", side_effect=AssertionError("Unexpected allocation")):
      with patch.object(island, "direct_dsu") as compute:
        island.island(SimpleNamespace(ntree=2), SimpleNamespace(nworld=3), parent=parent)
      self.assertIs(compute.call_args.args[2], parent)
      model = Mock(nactuator=2, opt=Mock(disableflags=0))
      with patch.object(wp, "launch", side_effect=RuntimeError("first launch")) as launch:
        with self.assertRaisesRegex(RuntimeError, "first launch"):
          derivative.deriv_smooth_vel(model, Mock(nworld=3), Mock(), actuator_vel=velocity)
      self.assertIs(launch.call_args.kwargs["outputs"][0], velocity)
      with patch.object(wp, "launch", side_effect=RuntimeError("first launch")) as launch:
        with self.assertRaisesRegex(RuntimeError, "first launch"):
          constraint.make_constraint(Mock(), Mock(nworld=3), efc_nnz=nonzeros)
      self.assertTrue(any(value is nonzeros for value in launch.call_args.kwargs["inputs"]))

  def test_stage_scratch_records_are_passive_and_workspace_has_one_resource_tree(self):
    records = (
      forward._PositionScratch,
      forward._ImplicitScratch,
      forward._ForwardScratch,
      forward._StepScratch,
      solver._SolverScratch,
      smooth._TransmissionScratch,
      forward.collision_driver._CollisionScratch,
    )
    for record in records:
      with self.subTest(record=record.__name__):
        self.assertTrue(dataclasses.is_dataclass(record))
        tree = ast.parse(inspect.getsource(record))
        self.assertFalse(any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in ast.walk(tree)))
        self.assertFalse({"bindings", "updates", "device", "storage", "capacity", "arrays"} & record.__annotations__.keys())
    self.assertFalse(
      {"convex", "_collision", "_solver_context", "_solver_model", "_solver_data"} & _StepWorkspace.__annotations__.keys()
    )
    self.assertIn("scratch", _StepWorkspace.__annotations__)

  def test_numerical_composition_passes_only_each_stages_resources(self):
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
    position = forward._PositionScratch(object(), object(), object(), object(), object())
    implicit = forward._ImplicitScratch(*(object() for _ in range(6)))
    with ExitStack() as stack:
      for module, name in (
        (forward, "fwd_kinematics"),
        (smooth, "crb"),
        (smooth, "tendon_armature"),
        (sleep, "wake_collision"),
        (sleep, "update_sleep"),
        (smooth, "factor_solve_i"),
        (forward, "_launch_implicit_free_body_solve"),
        (forward, "_advance"),
      ):
        stack.enter_context(patch.object(module, name))
      operations = [
        stack.enter_context(patch.object(module, name))
        for module, name in (
          (constraint, "make_constraint"),
          (island, "island"),
          (smooth, "transmission"),
          (derivative, "deriv_smooth_vel"),
        )
      ]
      forward.fwd_position(model, data, factorize=False, scratch=position)
      operations[0].assert_called_once_with(model, data, efc_nnz=position.efc_nnz)
      operations[1].assert_called_once_with(model, data, parent=position.island_parent)
      operations[2].assert_called_once_with(model, data, scratch=position.transmission)
      forward.implicit(model, data, scratch=implicit)
      operations[3].assert_called_once_with(model, data, implicit.qDeriv, actuator_vel=implicit.actuator_vel)

  def test_transmission_initializes_supplied_counters_on_every_execution(self):
    model, data = Mock(nacttrnbody=2), Mock(nworld=3)
    moment_nnz = wp.empty(3, dtype=int, device="cpu")
    body_ncon = wp.empty((3, 2), dtype=int, device="cpu")
    scratch = smooth._TransmissionScratch(moment_nnz, body_ncon)
    with patch.object(wp, "launch"), patch.object(wp, "empty", side_effect=AssertionError("Unexpected allocation")):
      for poison in (17, -91):
        moment_nnz.fill_(poison)
        body_ncon.fill_(poison)
        smooth.transmission(model, data, scratch=scratch)
        np.testing.assert_array_equal(moment_nnz.numpy(), 0)
        np.testing.assert_array_equal(body_ncon.numpy(), 0)

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
        patch.object(step_program, "validate_step_workspace"),
        patch.object(wp, "capture_get_memory_operations", return_value=()),
        patch.object(step_program.graph_ops, "adopt_launches", return_value=(object(),)) as adopt,
      ):
        if rank == 2:
          step_program.bind_step_program(_program_workspace(bindings), graph, (record,))
          adopt.assert_called_once()
          self.assertIs(adopt.call_args.kwargs["extents"][0][2], bindings.world_storage.protected_count)
        else:
          with self.assertRaisesRegex(ValueError, "no native count declaration"):
            step_program.bind_step_program(_program_workspace(bindings), graph, (record,))
          adopt.assert_not_called()
          self.assertTrue(bindings.recording_failed)
          self.assertTrue(graph._preparation_failed)

  def test_prepared_execution_binding_cannot_change_mode_count_or_bindings(self):
    """Keep the sole count in the owner record and reject mode or descriptor replacement."""
    bindings = _step_bindings()
    count = bindings.world_storage.protected_count
    specs = _minimal_scratch_specs()
    arrays = {spec.name: wp.empty(spec.shape, dtype=spec.dtype, device="cpu") for spec in specs}
    data = _workspace_data(count)
    with _cpu_workspace_preparation(specs), patch.object(native_workspace.field_ops, "lookup", return_value=0):
      dynamic = make_step_workspace(Metadata(), data, arrays=arrays, bindings=bindings)
      fixed = make_step_workspace(Metadata(), data, arrays=arrays)
    self.assertFalse(hasattr(dynamic, "world_live_count"))
    self.assertIs(dynamic.scratch.forward.solver.world_count, count)
    self.assertIsNone(fixed.scratch.forward.solver.world_count)
    for replacement in (None, _step_bindings()):
      dynamic.bindings = replacement
      with self.assertRaisesRegex(ValueError, "execution binding changed"):
        step_program.validate_step_workspace(dynamic, dynamic.model, data)
    dynamic.bindings = bindings
    count.fill_(1)
    step_program.validate_step_workspace(dynamic, dynamic.model, data)
    for name, changed in (("shape", (0,)), ("strides", (8,)), ("dtype", wp.int64)):
      original = getattr(count, name)
      try:
        setattr(count, name, changed)
        with self.assertRaisesRegex(ValueError, "storage or count descriptors changed"):
          step_program.validate_step_workspace(dynamic, dynamic.model, data)
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
          step_program.validate_step_workspace(dynamic, dynamic.model, data)
      finally:
        setattr(owner, name, original)
    step_program.validate_step_workspace(dynamic, dynamic.model, data)
    fixed.bindings = bindings
    with self.assertRaisesRegex(ValueError, "execution binding changed"):
      step_program.validate_step_workspace(fixed, fixed.model, data)

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
    before = step_program._layout(model), step_program._layout(template)
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
    self.assertEqual(before, (step_program._layout(model), step_program._layout(template)))

  def test_caller_scratch_binding_owns_no_allocation(self):
    specs = _minimal_scratch_specs()
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
      step_program.validate_step_workspace(workspace, workspace.model, data)

  def test_snapshot_aliases_share_one_traversal_but_later_mutations_remain_visible(self):
    array = wp.empty(2, dtype=wp.float32, device="cpu")
    metadata = Metadata(pair_counts=(array,))
    first = step_program._layout((metadata, metadata, array))
    self.assertIs(first[1][0], step_program._LAYOUT_REFERENCE)
    self.assertIs(first[2][0], step_program._LAYOUT_REFERENCE)
    self.assertNotEqual(first[1], first[2])
    self.assertEqual(first, step_program._layout((metadata, metadata, array)))
    split_alias = dataclasses.replace(metadata)
    self.assertNotEqual(first, step_program._layout((metadata, split_alias, array)))
    metadata.option += 1
    self.assertNotEqual(first, step_program._layout((metadata, metadata, array)))
    metadata.option -= 1
    array.shape = (1,)
    self.assertNotEqual(first, step_program._layout((metadata, metadata, array)))
    array.shape = (2,)
    self.assertEqual(first, step_program._layout((metadata, metadata, array)))

    # A short-lived traversal retains temporary roots, preventing recycled IDs.
    temporary = Metadata()
    reference = weakref.ref(temporary)
    memo = {}
    step_program._layout(temporary, memo)
    del temporary
    self.assertIsNotNone(reference())
    del memo
    self.assertIsNone(reference())

  def test_derived_scratch_record_rebinding_rejected_before_capture(self):
    specs = _minimal_scratch_specs()
    arrays = {spec.name: wp.empty(spec.shape, dtype=spec.dtype, device="cpu") for spec in specs}
    data = _workspace_data(arrays["collision_worldid"])
    with _cpu_workspace_preparation(specs):
      workspace = make_step_workspace(Metadata(), data, arrays=arrays)
    original = workspace.scratch
    for replacement in (
      dataclasses.replace(original, implicit=object()),
      dataclasses.replace(original, forward=dataclasses.replace(original.forward, solver=object())),
    ):
      workspace.scratch = replacement
      with self.assertRaisesRegex(ValueError, "scratch descriptors"):
        step_program.validate_step_workspace(workspace, workspace.model, data)
    workspace.scratch = original
    step_program.validate_step_workspace(workspace, workspace.model, data)

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
        _data_layout=step_program._layout(data),
        _model_layout=step_program._layout(model),
      )
      value = getattr(workspace, owner)
      setattr(value, field, getattr(value, field) + 1)
      with self.subTest(owner=owner, field=field), self.assertRaisesRegex(ValueError, "scalar metadata"):
        step_program.validate_step_workspace(workspace, model, data)

  def test_foreign_model_or_data_rejected(self):
    workspace = self.metadata_workspace
    for model, data in ((Metadata(), workspace.data), (workspace.model, Metadata())):
      with self.assertRaisesRegex(ValueError, "original model"):
        step_program.validate_step_workspace(workspace, model, data)

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
        _data_layout=step_program._layout(data),
        _model_layout=step_program._layout(model),
      )
      nested[0] = wp.ones(1, dtype=float, device="cpu")
      with self.subTest(owner=owner), self.assertRaisesRegex(ValueError, "scalar metadata"):
        step_program.validate_step_workspace(workspace, model, data)

  def test_prepared_topology_tuple_or_callback_change_rejected(self):
    for field, replacement in (("pair_counts", (2, 2)), ("callback", lambda *_: None)):
      workspace = dataclasses.replace(self.metadata_workspace, model=Metadata())
      setattr(workspace.model, field, replacement)
      with self.assertRaisesRegex(ValueError, "scalar metadata"):
        step_program.validate_step_workspace(workspace, workspace.model, workspace.data)

  def test_blocked_scratch_alignment_rejects_base_and_world_stride(self):
    """Reject misaligned blocked matrices without restricting dense scalar counters."""
    specs = (
      *_minimal_scratch_specs(),
      WorkspaceFieldSpec("solver.h", (2, 64, 64), wp.float32, "world"),
      WorkspaceFieldSpec("solver.hfactor", (2, 64, 64), wp.float32, "world"),
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
      *_minimal_scratch_specs(),
      WorkspaceFieldSpec("solver.h", (2, 2, 2), wp.float32, "world"),
      WorkspaceFieldSpec("solver.hfactor", (2, 0, 0), wp.float32, "world"),
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
