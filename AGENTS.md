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

- MJWarp owns native numerical fields, shape formulas, alignment, initialization, reset and state-transfer semantics. The composing engine owns storage selection, domain admission, symbolic source binding, graph publication and retirement. No production MJWarp module, public export, or package dependency may import GPU Components.
- Native numerical work consists of passive typed records and arrays. Keep SolverContext and CollisionContext as numerical operands; Data contains simulation state, never an allocator, registry, graph binding or ownership ledger. Do not recreate Data.scratch, ScratchRegistry, StepBindings, workspace.py, step_program.py or step_execution.py.
- Numerical stages own temporary arrays at their ordinary Warp allocation sites and initialize them on every execution before reading them. Internal numerical contexts may group operands used together by an algorithm; they are created locally, never injected by an external storage backend. Do not restate allocations in work layouts, scratch registries, forwarding parameters or schema lookup tables. The composing engine owns recorded lifetimes and storage bindings.
- Keep one numerical execution path. Do not branch on storage backends or duplicate physics orchestration in a composing engine. Standalone forward/step1/step2 allocate only temporaries reached by their numerical operations.
- When a replacement becomes the supported integration path, migrate every affected caller and delete superseded records, catalogs, adapters and optional injection parameters in the same change. A successful pilot is not a completed migration. Audit the whole package and gate the removed structure before declaring completion.
- Native array enumeration/rebinding and metadata signatures belong to io.array_fields, io.replace_arrays and io.metadata_signature. Do not recreate native schema walkers in consumers. Borrowed numerical context references use explicit field metadata so enumeration does not claim referenced Model/Data payload as temporary storage.
- Numerical stages use ordinary wp.launch, wp.launch_tiled, array zero_/fill_ and wp.copy. Explicit logical extents and scalar count operands preserve intent; equal bounds do not identify domains. Never infer intended work from raw shape equality, kernel identity, backing capacity or capture order.
- Keep native layout validation separate from the engine's supported-feature admission policy and generic storage/dependency proofs. Native metadata signatures cover concrete and symbolic descriptors, aliases and scalar options; callers validate around application callbacks and before publication.
- Neutral backing defaults are not episode reset semantics. Keep contact reference sentinels and matrix alignment in native operations; preserve model-dependent initialization, warm starts and contact reference remapping in their native lifecycle operations.
- Architecture tests reject forbidden files, registry/binding symbols, GPU Components imports/metadata dependencies, duplicate native shape/default catalogs and state arrays appearing as work ownership. Audit each file's existence and dependency direction before numerical validation.
- Native schema traversal releases borrowed payloads without cyclic collection. Recursive algorithms pass borrowed bindings explicitly; never capture them in a self-referential local function or persist traversal memoization across validations.
- Numerical island factoring requires a fresh structural partition of the current inertia and constraint support. Full stages provide that ordering through an explicit numerical option; standalone solve defaults must not infer freshness from active-DOF reuse or retained scratch. Reuse existing island CSR, never infer independence from observed zero Hessian entries. A new solve always refactors before cached factors may be reused.
- Initialize physical quantities according to their equations, independently of compact/sparse storage flags. Test first-write behavior with poisoned storage and reject nonfinite values in numerical parity checks.
- Cooperative stages must cover every tile entry even when `launch_tiled` uses one CPU lane. Use block-uniform loop bounds and masked tile accesses so every lane reaches collective barriers; do not assume a lane exists for every logical entry.

- Line length limit is 128 characters. Docstring length limit is 100 characters.
- Avoid formatting-only edits in integration diffs. Preserve the original layout of unchanged numerical calls when it fits the line limit. Remove trailing commas left by deleted wrappers when they force unnecessary argument-per-line formatting; manually review formatter output.
- Prefer targeted, efficient tests over exhaustive edge-case coverage.
