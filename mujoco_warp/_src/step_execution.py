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

"""Native step count semantics and explicit recording operations.

Workspace preparation and physics stages depend on this module. It owns no solver,
collision algorithm or scratch schema; generic graph and field operations own the
underlying CUDA mechanics.
"""

import dataclasses

import warp as wp
from gpu_components import fields as field_ops
from gpu_components import graph as graph_ops
from gpu_components.field_data import FieldStorage
from gpu_components.graph_data import GraphKernelBinding
from gpu_components.graph_data import GraphUpdateTable
from gpu_components.graph_data import KernelParameterBinding


@dataclasses.dataclass(eq=False)
class StepBindings:
  """Borrowed native capacity domains and their single captured binding ledger.

  Keep all three storage owners alive through graph retirement. World launches use
  world_storage.protected_count; candidate and CCD launches use their ready_count.
  Admission must keep those counts within physically ready rows. Set updates before
  the first recorded operation. Only native binding operations mutate the ledgers
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
  operations: list[dict] = dataclasses.field(default_factory=list)
  _recording_binding: tuple | None = None


_LAYOUT_REFERENCE = object()


def _layout(value, memo=None):
  """Snapshot native metadata with local references for repeated borrowed aliases.

  Share a fresh memo across one operation's roots. First occurrences record all
  descriptors; later occurrences record their traversal ordinal. The memo retains
  temporary containers and is never reused by a subsequent validation.
  """
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
  if _layout(data, memo) != workspace._data_layout or _layout(model, memo) != workspace._model_layout:
    raise ValueError("Prepared Model/Data descriptors or scalar metadata changed; prepare a new workspace")
  if (
    _layout(
      (
        tuple(workspace.arrays.items()),
        workspace.convex,
        workspace._collision,
        workspace._solver_context,
        workspace._solver_model,
        workspace._solver_data,
      ),
      memo,
    )
    != workspace._scratch_layout
  ):
    raise ValueError("Prepared scratch descriptors changed; prepare a new workspace")
  owner = graph_ops.current_capture(device=workspace.device)
  if owner is not None:
    try:
      if workspace.bindings is not None:
        for storage in (workspace.bindings.world_storage, workspace.bindings.contact_storage, workspace.bindings.ccd_storage):
          field_ops.retain_graph(storage, owner)
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
  ) + (id(bindings.bindings), id(bindings.operations))


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
    raise ValueError("Native recording binding changed; restore its original storage, counts, updates and ledgers")


def _resolve_launch_counts(bindings, kernel, extent_domain, extent_axis, parameter_domains):
  """Resolve native count declarations before emitting a launch; mutate no recording state.

  A dynamic extent is the leading axis of world, candidate or CCD storage. Fixed
  worker grids explicitly use extent_domain=None and extent_axis=None. Named int32
  scalar arguments may independently use any of those domains. Returned descriptors
  borrow their exact count arrays; parameter indices include Warp's launch bounds.
  """
  _validate_bindings(bindings)
  owners = {"world": bindings.world_storage, "candidate": bindings.contact_storage, "ccd": bindings.ccd_storage}
  if extent_domain is not None and extent_domain not in owners:
    raise ValueError(f"Unknown native extent domain: {extent_domain!r}")
  if extent_domain is None and extent_axis is not None:
    raise ValueError("A fixed worker grid must explicitly omit its extent axis")
  if extent_domain is not None and (type(extent_axis) is not int or extent_axis != 0):
    raise ValueError("Native dynamic extents require explicit leading axis zero")
  if parameter_domains is not None and len(parameter_domains) > 4:
    raise ValueError("A native launch supports at most four count parameters")
  sources = {
    name: (owner.protected_count if name == "world" else owner.ready_count, owner.capacity) for name, owner in owners.items()
  }
  labels = {argument.label: (index + 1, argument.type) for index, argument in enumerate(kernel.adj.args)}
  parameters = []
  for name, domain in (parameter_domains or {}).items():
    if name not in labels or domain not in sources:
      raise ValueError(f"Unknown native count argument or domain: {name!r}, {domain!r}")
    index, dtype = labels[name]
    if dtype not in (int, wp.int32):
      raise ValueError(f"Native count argument must be int32: {name!r}")
    parameters.append(KernelParameterBinding(index, *sources[domain]))
  return sources[extent_domain][0] if extent_domain is not None else None, tuple(parameters)


def _begin_recording(bindings, stream=None):
  if bindings.updates is None:
    raise RuntimeError("Prepare native graph updates before recording step operations")
  device = bindings.world_storage.device
  graph = graph_ops.current_capture(device=device, stream=stream)
  if graph is None:
    raise RuntimeError("Record native step operations inside a Warp-managed graph capture")
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


def _record_launch(bindings, binding, kernel, dim, extent_domain, parameter_domains):
  bindings.bindings.append(binding)
  bindings.operations.append(
    {
      "operation": "launch",
      "kernel": kernel.key,
      "module": kernel.func.__module__,
      "function": kernel.func.__qualname__,
      "dim": list((dim,) if isinstance(dim, int) else dim),
      "extent_domain": extent_domain,
      "extent_axis": binding.extent_axis,
      "parameter_domains": dict(parameter_domains or {}),
      "launch_rank": binding.launch_rank,
      "node": binding.node,
    }
  )


def launch_step_kernel(
  bindings,
  kernel,
  dim,
  *,
  inputs=None,
  outputs=None,
  extent_domain,
  extent_axis=0,
  parameter_domains=None,
  tiled=False,
  block_dim=0,
  max_blocks=0,
  device=None,
  stream=None,
):
  """Emit one native kernel and its declared count binding as one operation.

  Fixed execution supplies bindings=None. Dynamic execution resolves native count
  semantics before emission, then delegates launch and node ownership to the graph
  component. Native diagnostics use the existing StepBindings ledger. No launch is
  inferred from numeric dimensions. A failed dispatch invalidates the recording.
  """
  if bindings is None:
    return graph_ops.launch(
      None,
      kernel,
      dim,
      inputs=inputs,
      outputs=outputs,
      tiled=tiled,
      block_dim=block_dim,
      max_blocks=max_blocks,
      device=device,
      stream=stream,
    )
  extent, parameters = _resolve_launch_counts(bindings, kernel, extent_domain, extent_axis, parameter_domains)
  if bindings.updates is None:
    raise RuntimeError("Prepare native graph updates before recording step operations")
  if type(tiled) is not bool or (tiled and (type(block_dim) is not int or block_dim < 1)):
    raise ValueError("Tiled recording requires an explicit positive block dimension")
  try:
    binding = graph_ops.launch(
      bindings.updates,
      kernel,
      dim,
      inputs=inputs,
      outputs=outputs,
      extent_axis=extent_axis,
      extent_source=extent,
      parameters=parameters,
      tiled=tiled,
      block_dim=block_dim,
      max_blocks=max_blocks,
      device=device,
      stream=stream,
    )
    if binding is not None:
      _begin_recording(bindings, stream)
      _record_launch(bindings, binding, kernel, dim, extent_domain, parameter_domains)
    return binding
  except BaseException:
    _fail_recording(bindings, stream)
    raise


def fill_step_rows(bindings, array, value, domain):
  """Fill native storage rows using that domain's independent admitted count."""
  if bindings is None:
    if array.size:
      array.fill_(value)
    return array
  try:
    _validate_bindings(bindings)
    owners = {"world": bindings.world_storage, "candidate": bindings.contact_storage, "ccd": bindings.ccd_storage}
    if domain not in owners:
      raise ValueError(f"Unknown native memory domain: {domain!r}")
    owner = owners[domain]
    if not isinstance(array, wp.array):
      raise ValueError("Native fill requires a Warp array")
    if not array.size:
      field_ops.validate_array(array, dtype=array.dtype, shape=array.shape, device=owner.device)
      if array.shape[0] and field_ops.lookup(owner, array) is None:
        raise ValueError("Native fill requires a registered field or absent-leading-axis sentinel")
      return array
    _begin_recording(bindings)
    field_ops.fill(owner, array, value, count=owner.protected_count if domain == "world" else owner.ready_count)
    bindings.operations.append({"operation": "fill", "domain": domain, "field": field_ops.lookup(owner, array).name})
    return array
  except BaseException:
    _fail_recording(bindings)
    raise


def copy_step_rows(bindings, destination, source, domain):
  """Copy native storage rows without accessing the inactive capacity suffix."""
  if bindings is None:
    if destination.size:
      wp.copy(destination, source)
    return
  try:
    _validate_bindings(bindings)
    owners = {"world": bindings.world_storage, "candidate": bindings.contact_storage, "ccd": bindings.ccd_storage}
    if domain not in owners:
      raise ValueError(f"Unknown native memory domain: {domain!r}")
    owner = owners[domain]
    if not isinstance(destination, wp.array):
      raise ValueError("Native copy requires Warp arrays")
    if not destination.size:
      field_ops.validate_array(destination, dtype=destination.dtype, shape=destination.shape, device=owner.device)
      field_ops.validate_array(source, dtype=destination.dtype, shape=destination.shape, device=owner.device)
      if destination.shape[0] and field_ops.lookup(owner, destination) is None and field_ops.lookup(owner, source) is None:
        raise ValueError("Native copy requires a registered field or absent-leading-axis sentinel")
      return
    _begin_recording(bindings)
    field_ops.copy(owner, destination, source, count=owner.protected_count if domain == "world" else owner.ready_count)
    field = field_ops.lookup(owner, destination) or field_ops.lookup(owner, source)
    bindings.operations.append({"operation": "copy", "domain": domain, "field": field.name})
  except BaseException:
    _fail_recording(bindings)
    raise
