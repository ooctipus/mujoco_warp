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

- Prepared execution must keep live counts and launch bindings coherent: dynamic workspaces require both a world-live-count scalar and the bounded launch/copy/fill recorder; fixed workspaces supply neither. Reject incomplete bindings before recording.
- Schema field names belong to preparation. Numeric identity and membership APIs must not also resolve paths or names.
- Name allocation specifications separately from allocated views. A recorder binds already-emitted launches; its parameter-domain declarations are not kernel argument values.
- Initialize physical quantities according to their equations, independently of compact/sparse storage flags. Test first-write behavior with poisoned storage and reject nonfinite values in numerical parity checks.

- Line length limit is 128 characters. Docstring length limit is 100 characters.
- Prefer targeted, efficient tests over exhaustive edge-case coverage.
