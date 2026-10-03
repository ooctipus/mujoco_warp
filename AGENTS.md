# Development Workflow

- Always use `uv run`, not python.
- Always run `uv run pytest -n 8` before creating a PR.
- Run `uv run pre-commit install` after cloning to enable pre-commit hooks (ruff, uv-lock, kernel-analyzer).
- Prefer running individual tests rather than the full test suite to improve iteration speed.

# Commits and PRs

- PR body should be plain, concise prose. Describe the problem, what the change does, and any non-obvious tradeoffs. Bullet points listing changes are fine, but avoid section headers, structured templates, and emojis. A good PR description reads like a short paragraph to a colleague, not a form.
- PR and commit messages are rendered on GitHub, so don't hard-wrap them at 88 columns. Let each sentence flow on one line.
- Push branches to your own fork, not to the google-deepmind/mujoco_warp repo directly.
- Do not add AI assistants (e.g. Claude) as commit co-authors. The mujoco_warp CLA check requires all commit authors to have signed the CLA, and AI tools cannot sign it.
- Amending commits is fine before a PR has reviewers looking at it. Once a PR is under review, use new commits so reviewers can see what changed.
- When responding to PR review comments:
  - Reply to each comment individually confirming what you did (or why you didn't).
  - Resolve comment threads that are addressed.
  - Add a summary comment on the PR after responding, covering what was applied and what was intentionally skipped.

# Code Style

- Prepared execution borrows one passive native StepBindings record; its world storage protected count is the sole world-live-count source. Fixed workspaces supply no bindings. The native program composition boundary owns domain declarations and one binding/failure ledger, calling gpu-components directly; reject callback recorders and duplicate count arguments.
- Schema field names belong to preparation. Numeric identity and membership APIs must not also resolve paths or names.
- Prepared resource validation, StepBindings, numeric count resolution and captured-program composition belong to `step_program.py`. Workspace preparation depends on it; numerical modules depend on neither program nor workspace modules. Do not recreate `step_execution.py` or compatibility aliases. Keep stage scratch schemas beside their numerical stages and export public program operations from their canonical owner.
- Numerical stages use ordinary `wp.launch`, `wp.launch_tiled`, array `zero_`/`fill_` and `wp.copy`. Custom Warp records their actual launches and memory operations. Do not add native dispatch or memory wrappers, monkeypatches, ambient contexts, callbacks or execution bindings to numerical code.
- Preserve original signatures for computations needing no scratch. A stage needing scratch borrows explicitly typed arrays or a passive stage-specific record containing only its numerical operands. Bind native collision scratch once at provider selection; external callbacks retain their original signatures.
- Prepared workspace and stage scratch records are passive dataclasses with no allocation, validation, execution, or report methods. Canonical convex annotations own dtype/rank; its shape function owns dimensions, and workspace preparation declares count domains. Do not repeat element types in shape tuples or route whole workspace objects into adopted convex kernels.
- Use gpu-components layout/pack/bind/validation operations for byte representation, and public graph current_capture/retain/invalidate operations for shared capture metadata. Native code must not read or write generic graph-private ownership/failure attributes. Solver-specific alignment checks belong to solver operations, never scratch-name branches in the workspace allocator.
- Dynamic workspaces require borrowed scratch registered with each explicitly declared FieldStorage; reject missing backing before allocation. Optional zero-leading-axis sentinels have no payload and need no owner registration. Global counters remain dense and reset completely every execution.
- Name allocation specifications separately from allocated views. Carry execution counts as Warp operands at each numerical use; resolve parameter identities to device scalars at the composition boundary. Do not maintain a second kernel/factory-to-count catalog or factory provenance for binding. Ordinary kernel memoization remains valid. Captured programs remain unpublished until complete validation and binding; unresolved operands, failed adoption and missing dependency proofs reject publication. Freeze borrowed source/updater identity during recording and retain bindings through graph retirement.
- Unsupported prepared features retain their eager behavior and are rejected at workspace admission before recording. Composing owners validate prepared metadata before numerical execution and after application callbacks; do not push ownership checks into numerical stages or cache validation across calls.
- Native array enumeration and rebinding belong to `io.array_fields` and `io.replace_arrays`, using existing annotations. Consumers must not recreate native schema walkers. Workspace owns one scratch tree; do not keep duplicate derived-view aliases. Each stage has exactly one source for each scratch operand.
- Native schema traversal must release borrowed payloads when its result is dropped, without requiring cyclic collection. Recursive algorithms pass borrowed bindings explicitly; never capture them in a self-referential local function.
- Choose eager or borrowed scratch in the stage operation, keeping required initialization on every execution. Merge allocation-only wrapper pairs into one operation with optional scratch. Do not introduce validation bypass flags, cross-call caches, generic execution contexts or one-use forwarding helpers. Preserve event labels at the numerical body.
- Snapshot borrowed descriptor aliases once per preparation/validation operation, retaining temporary roots for that traversal. Never persist traversal memoization across validations; later scalar and array-descriptor mutations must remain visible.
- Borrowed scratch validates supplied descriptors without planning a hypothetical allocation. Retain immutable field specifications for reports instead of duplicating their metadata into dictionaries. Native recording retains actual graph bindings, not a parallel diagnostic operation history.
- Compile all native records in one numeric adoption batch from canonical operand occurrences. Memory operations declare their logical extents independently of storage readiness; validate exact registered fields and explicitly dense global operands. Omitted extents retain whole-array meaning. Never infer intended work from shape, kernel identity, storage capacity or capture order. Unregistered operands and missing dependency proofs reject publication. Warp and gpu-components own generic byte semantics, graph identity and dependency validation.
- Workspace preparation borrows the original concrete Data and owns its shallow execution projection. Freeze both descriptors together and validate the original Data after callbacks; do not create a second projection in a composing engine. Preparation resolves symbolic counts to upper bounds for concrete layouts only. Borrowed execution descriptors and compact aliases retain parameter identity; equal maxima do not make two parameters interchangeable. Concrete-only IO rejects symbolic descriptors before reading device data. Keep the solver's world count in its Data operand, without a second scratch count source.
- Numerical island factoring requires a fresh structural partition of the current inertia and constraint support. Full stages provide that ordering through an explicit numerical option; standalone solve defaults must not infer freshness from active-DOF reuse or retained scratch. Reuse existing island CSR, never infer independence from observed zero Hessian entries. A new solve always refactors before cached factors may be reused.
- Initialize physical quantities according to their equations, independently of compact/sparse storage flags. Test first-write behavior with poisoned storage and reject nonfinite values in numerical parity checks.
- Cooperative stages must cover every tile entry even when `launch_tiled` uses one CPU lane. Use block-uniform loop bounds and masked tile accesses so every lane reaches collective barriers; do not assume a lane exists for every logical entry.

- Line length limit is 128 characters. Docstring length limit is 100 characters.
- Avoid formatting-only edits in integration diffs. Preserve the original layout of unchanged numerical calls when it fits the line limit. Remove trailing commas left by deleted wrappers when they force unnecessary argument-per-line formatting; manually review formatter output.
- Prefer targeted, efficient tests over exhaustive edge-case coverage.
