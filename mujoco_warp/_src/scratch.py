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
"""Scratch temporaries for numerical stages.

``scratch_array(d.scratch, name, shape, dtype)`` replaces ``wp.empty(shape, dtype)``
one for one. With ``d.scratch is None`` it is that allocation. Under a prepared
step the composition owns a registry: the stage's requests were recorded once
and are answered by name from prepared storage. The stage does not know which.
"""

from gpu_components.scratch import scratch_array as scratch_array

__all__ = ["scratch_array"]
