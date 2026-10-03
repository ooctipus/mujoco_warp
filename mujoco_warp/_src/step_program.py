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

"""Bind numerical count operands to the composing engine's admitted device scalars.

Warp preserves count identity at each operation. This boundary validates native
storage and passes that numeric relation to GPU Components; it does not classify
kernels or reconstruct count meaning from code identity, shapes or launch order.
"""

from __future__ import annotations

import dataclasses

import warp as wp
from gpu_components import fields as field_ops
from gpu_components import graph as graph_ops
from gpu_components.field_data import FieldStorage
from gpu_components.graph_data import GraphKernelBinding
from gpu_components.graph_data import GraphUpdateTable


@dataclasses.dataclass(eq=False)
class StepBindings:
  """Borrowed native capacity domains and their single captured binding ledger.

  Keep all three storage owners alive through graph retirement. World launches use
  world_storage.protected_count; candidate and CCD launches use their ready_count.
  Admission must keep those counts within physically ready rows. Set updates before
  the first recorded operation. Only native binding operations mutate the bindings
  and recording snapshot; do not replace their storage or count sources. A composing
  engine also sets recording_failed on application or capture failure. Never clear
  this latch or publish a graph after it becomes true.
  """

  world_storage: FieldStorage
  contact_storage: FieldStorage
  ccd_storage: FieldStorage
  updates: GraphUpdateTable | None = None
  recording_failed: bool = False
  bindings: list[GraphKernelBinding] = dataclasses.field(default_factory=list)
  _recording_binding: tuple | None = None


_LAYOUT_REFERENCE = object()


def _layout(value, memo=None):
  """Snapshot native metadata with local references for repeated borrowed aliases.

  Share a fresh memo across one operation's roots. First occurrences record all
  descriptors; later occurrences record their traversal ordinal. The memo retains
  temporary containers and is never reused by a subsequent validation.
  """
  if isinstance(value, wp.CountParameter):
    return wp.CountParameter, id(value), value.maximum
  if value is None or isinstance(value, (int, float, bool, str)):
    return value
  if memo is None:
    memo = {}
  identity = id(value)
  if identity in memo:
    return memo[identity][1]
  if isinstance(value, wp.array):
    memo[identity] = value, (_LAYOUT_REFERENCE, len(memo))
    return value.ptr, value.shape, value.strides, value.dtype, value.device
  if dataclasses.is_dataclass(value):
    memo[identity] = value, (_LAYOUT_REFERENCE, len(memo))
    result = []
    for field in dataclasses.fields(value):
      child = getattr(value, field.name)
      cached = memo.get(id(child))
      result.append((field.name, cached[1] if cached is not None else _layout(child, memo)))
    return tuple(result)
  if isinstance(value, (tuple, list)):
    memo[identity] = value, (_LAYOUT_REFERENCE, len(memo))
    result = []
    for child in value:
      cached = memo.get(id(child))
      result.append(cached[1] if cached is not None else _layout(child, memo))
    return tuple(result)
  return identity


def validate_step_workspace(workspace, model, data):
  """Validate prepared binding before emission and pin owners during graph recording."""
  if workspace.bindings is not workspace._execution_binding:
    raise ValueError("Prepared execution binding changed; restore its original bindings before recording")
  if workspace.bindings is not None:
    if _binding_layout(workspace.bindings) != workspace._binding_layout:
      raise ValueError("Prepared native storage or count descriptors changed; prepare a new workspace")
    _validate_bindings(workspace.bindings)
  if model is not workspace.model or data is not workspace.data:
    raise ValueError("Prepared workspace requires its original model, data and immutable step options")
  memo = {}
  if (
    _layout((data, workspace.execution_data), memo) != workspace._data_layout or _layout(model, memo) != workspace._model_layout
  ):
    raise ValueError("Prepared Model/Data descriptors or scalar metadata changed; prepare a new workspace")
  if (
    _layout(
      (tuple(workspace.arrays.items()), workspace.scratch),
      memo,
    )
    != workspace._scratch_layout
  ):
    raise ValueError("Prepared scratch descriptors changed; prepare a new workspace")
  owner = graph_ops.current_capture(device=workspace.device)
  if owner is not None:
    try:
      if workspace.bindings is not None:
        if workspace.bindings.updates is None:
          raise RuntimeError("Prepare native graph updates before recording step operations")
        _retain_bindings(workspace.bindings, owner)
      graph_ops.retain(owner, workspace)
    except BaseException:
      if workspace.bindings is not None:
        workspace.bindings.recording_failed = True
      graph_ops.invalidate(owner)
      raise


def _binding_layout(bindings):
  if bindings is None:
    return None
  return tuple(
    (id(storage), storage.capacity, id(storage.device), id(count), (count.ptr, count.shape, count.strides, count.dtype))
    for storage, count in (
      (bindings.world_storage, bindings.world_storage.protected_count),
      (bindings.contact_storage, bindings.contact_storage.ready_count),
      (bindings.ccd_storage, bindings.ccd_storage.ready_count),
    )
  ) + (id(bindings.bindings),)


def _validate_bindings(bindings):
  if not isinstance(bindings, StepBindings):
    raise TypeError("Prepared bindings must be a native StepBindings record")
  if bindings.recording_failed:
    raise RuntimeError("The native step program has a failed recording")
  if not all(
    isinstance(storage, FieldStorage) for storage in (bindings.world_storage, bindings.contact_storage, bindings.ccd_storage)
  ):
    raise TypeError("Native capacity domains must borrow FieldStorage records")
  device = bindings.world_storage.device
  for storage, count in (
    (bindings.world_storage, bindings.world_storage.protected_count),
    (bindings.contact_storage, bindings.contact_storage.ready_count),
    (bindings.ccd_storage, bindings.ccd_storage.ready_count),
  ):
    if storage.closed or storage.service_failed:
      raise RuntimeError("Native step storage is closed or quarantined")
    if type(storage.capacity) is not int or not 0 < storage.capacity <= 2**31 - 1:
      raise ValueError("Native storage capacity must be a positive int32 count")
    if (
      storage.device != device
      or not isinstance(count, wp.array)
      or count.device != device
      or count.dtype != wp.int32
      or count.shape != (1,)
      or not count.is_contiguous
    ):
      raise ValueError("Native counts must be contiguous int32 scalars on the storage device")
  updates = bindings.updates
  if updates is not None and (
    not isinstance(updates, GraphUpdateTable)
    or updates.device != device
    or updates.enable_count is not bindings.world_storage.protected_count
    or updates.enable_count_maximum != bindings.world_storage.capacity
  ):
    raise ValueError("Native updates must use the exact world protected count and capacity")
  if bindings._recording_binding is not None and bindings._recording_binding != (_binding_layout(bindings), id(updates)):
    raise ValueError("Native recording binding changed; restore its original storage, counts, updates and bindings")


def _retain_bindings(bindings, graph):
  """Retain admitted resources and freeze the one binding relation for this program."""
  _validate_bindings(bindings)
  if bindings._recording_binding is None:
    graph_ops.retain(graph, bindings)
    for storage in (bindings.world_storage, bindings.contact_storage, bindings.ccd_storage):
      field_ops.retain_graph(storage, graph)
    bindings._recording_binding = _binding_layout(bindings), id(bindings.updates)


def _fail_recording(bindings, stream=None):
  if not isinstance(bindings, StepBindings):
    return
  bindings.recording_failed = True
  try:
    device = (
      bindings.updates.device
      if isinstance(bindings.updates, GraphUpdateTable)
      else bindings.world_storage.protected_count.device
    )
    graph = graph_ops.current_capture(device=device, stream=stream)
    if graph is not None:
      graph_ops.invalidate(graph)
  except BaseException:
    # A malformed borrowed descriptor must not mask the error that poisoned it.
    # The composition root must also reject the native recording_failed latch.
    pass


def bind_step_program(workspace, graph: wp.Graph, launches: tuple[wp.CapturedLaunch, ...]) -> None:
  """Bind exact numerical occurrences through their prepared count operands.

  The caller selects this population's records, excluding application callbacks.
  Memory records carry the intended region; registered storage only certifies its
  admission. Fixed operations have neither parameterized extents nor scalar uses.
  GPU Components proves record identity, operand coverage and updater ordering.
  """
  bindings = workspace.bindings
  try:
    validate_step_workspace(workspace, workspace.model, workspace.data)
    if bindings is None or bindings.updates is None:
      raise ValueError("Prepare native graph updates before adopting a step")
    if type(launches) is not tuple:
      raise TypeError("Native program launches must be an explicit immutable tuple")
    _retain_bindings(bindings, graph)
    storages = (bindings.world_storage, bindings.contact_storage, bindings.ccd_storage)
    count_sources = (
      (workspace.execution_data.nworld, bindings.world_storage.protected_count),
      (workspace.execution_data.naconmax, bindings.contact_storage.ready_count),
      (workspace.execution_data.naccdmax, bindings.ccd_storage.ready_count),
    )
    fixed_arrays = [workspace.data.nacon, workspace.data.ncollision]
    fixed_arrays.extend(workspace.arrays[spec.name] for spec in workspace._specs if spec.capacity_domain == "global_counter")
    fixed_arrays = tuple({(id(a.device), a.ptr, a.shape, a.strides, a.dtype): a for a in fixed_arrays}.values())
    launch_ids = {id(record) for record in launches}
    operations = tuple(op for op in wp.capture_get_memory_operations(graph) if id(op.launch) in launch_ids)
    field_ops.validate_memory_operations(graph, operations, storages, count_sources=count_sources, fixed_arrays=fixed_arrays)
    fixed = tuple(
      index for index, record in enumerate(launches) if not record.extent_parameters and not record.scalar_parameters
    )
    bindings.bindings.extend(
      graph_ops.adopt_launches(bindings.updates, graph, launches, count_sources=count_sources, fixed=fixed)
    )
  except BaseException:
    _fail_recording(bindings)
    graph_ops.invalidate(graph)
    raise
