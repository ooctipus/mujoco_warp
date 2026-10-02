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
- Native resource validation and bounded memory operations belong to `step_execution.py`; workspace preparation and physics stages depend on it. This leaf module must not import workspace, solver or collision modules, including through deferred imports. Postcapture native count composition belongs to `step_program.py`, which may import the physics definitions. Keep stage scratch schemas with their stages and public execution exports at their canonical owner; do not add workspace compatibility aliases.
- For adopted native stages, choose allocated or borrowed scratch once per stage from one shared layout; physics reads passive scratch fields. Physics uses ordinary `wp.launch` and `wp.launch_tiled`. Compose exact captured launch records with native domain declarations after capture; keep domain meanings in MJWarp and generic node ownership and dependency proofs in gpu-components. Do not thread execution-only bindings or workspaces through numerical stages.
- Prepared workspace and stage scratch records are passive dataclasses with no allocation, validation, execution, or report methods. Canonical convex annotations own dtype/rank; its shape function owns dimensions, and workspace preparation declares count domains. Do not repeat element types in shape tuples or route whole workspace objects into adopted convex kernels.
- Use gpu-components layout/pack/bind/validation operations for byte representation, and public graph current_capture/retain/invalidate operations for shared capture metadata. Native code must not read or write generic graph-private ownership/failure attributes. Solver-specific alignment checks belong to solver operations, never scratch-name branches in the workspace allocator.
- Dynamic workspaces require borrowed scratch registered with each explicitly declared FieldStorage; reject missing backing before allocation. Optional zero-leading-axis sentinels have no payload and need no owner registration. Global counters remain dense and reset completely every execution.
- Name allocation specifications separately from allocated views. Declare kernel operands at their plain Warp launches and native count meanings once in the program catalog. Match kernels and factory instances by exact identity, never names, dimensions or capture order. Clearing factory memoization must preserve weak provenance for still-live kernels; callers use the public cache-clear operation, never replace the registry. Resolve schema labels to numeric count bindings at the composition boundary. Captured programs remain unpublished until complete validation and binding; unknown launches, failed adoption and missing dependency proofs reject publication. Freeze borrowed source/updater identity during recording and retain bindings through graph retirement.
- Unsupported prepared features keep their original eager signatures. Reject them at workspace admission, rather than forwarding a permanently absent workspace through sensors, explicit integrators or other unsupported stages. Supported public stages validate their own prepared entry; do not cache validation across calls.
- Native array enumeration and rebinding belong to `io.array_fields` and `io.replace_arrays`, using existing type annotations. Consumers must not recreate native schema walkers. Each stage has one scratch source; do not accept both a workspace and separate overrides for fields it owns.
- Shared numerical stages consume explicit scratch arrays, never whole workspaces. Bindings are needed only by operations that touch bounded memory; kernel dispatch alone needs no execution operand. Public standalone entries validate and select or allocate resources; prepared composition entries validate and pass their borrowed operands directly. Keep required initialization in the numerical operation on every execution. Do not introduce validation bypass flags, cross-call validation caches, generic execution contexts or one-use forwarding helpers. Preserve event labels at the shared numerical body.
- Snapshot borrowed descriptor aliases once per preparation/validation operation, retaining temporary roots for that traversal. Never persist traversal memoization across validations; later scalar and array-descriptor mutations must remain visible.
- Borrowed scratch validates supplied descriptors without planning a hypothetical allocation. Retain immutable field specifications for reports instead of duplicating their metadata into dictionaries. Native recording retains actual graph bindings, not a parallel diagnostic operation history.
- Use the existing count-aware fill/copy operations for both eager and prepared execution; do not branch around their supported `bindings=None` path. Preserve Warp's eager `zero_()` fast path for integer zero without changing other fill values such as negative floating zero. Do not introduce native launch wrappers or generic dispatch interception. Native program compilation must validate field-operation count and storage descriptors; an internal kernel name alone is not an ownership proof. Preserve allocation timing and public entry validation.
- Initialize physical quantities according to their equations, independently of compact/sparse storage flags. Test first-write behavior with poisoned storage and reject nonfinite values in numerical parity checks.

- Line length limit is 128 characters. Docstring length limit is 100 characters.
- Prefer targeted, efficient tests over exhaustive edge-case coverage.
