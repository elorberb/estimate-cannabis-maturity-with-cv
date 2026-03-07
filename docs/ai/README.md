# AI Agent Context — Layout

This repo follows common practices for organizing agent instructions and Cursor rules.

## Root files

| File | Purpose |
|------|--------|
| **AGENTS.md** | Single source of truth for Python style, conventions, commands, and testing. Used by Cursor, Claude, and other tools. Keep at repo root. |
| **CLAUDE.md** | Short project context for Claude: what the repo does, pointer to AGENTS.md, and quick reference. Referenced by the GitHub Actions code-review workflow. |

## Cursor rules (`.cursor/rules/`)

Flat `.mdc` files with YAML frontmatter. Cursor picks these up automatically.

| Rule | Activation | Purpose |
|------|------------|--------|
| `project-python.mdc` | Always | Points to AGENTS.md for all Python and project conventions. |
| `skill-writing-plans.mdc` | When relevant | Use when you have a spec or multi-step task before writing code. |
| `skill-requesting-code-review.mdc` | When relevant | Use when completing work or before merge to request review. |
| `skill-receiving-code-review.mdc` | When relevant | Use when applying review feedback (verify first, then implement). |
| `skill-test-driven-development.mdc` | When relevant | Use before implementing features or bugfixes (test first, then code). |
| `skill-systematic-debugging.mdc` | When relevant | Use when debugging (find root cause before proposing fixes). |

- **Always-apply:** Only `project-python.mdc` (so AGENTS.md is always in context).
- **Apply intelligently:** Skill rules use `description` + `alwaysApply: false` so Cursor includes them when the user’s request matches.

## Best practices (from research)

- **One source of truth:** Agent conventions live in AGENTS.md; Cursor rules reference it instead of duplicating.
- **Version control:** All of this is committed so the whole team and CI use the same context.
- **Short CLAUDE.md:** Project overview and pointer to AGENTS.md; detailed rules stay in AGENTS.md and `.cursor/rules/`.
- **Flat rules:** `.cursor/rules/` is flat (no nested folders) for simplicity.

## Adding or changing rules

- New Cursor rule: add a `.mdc` file under `.cursor/rules/` with `description` and optionally `globs` / `alwaysApply`.
- New convention: update AGENTS.md (and CLAUDE.md if the project summary changes).
- New skill: add `skill-<name>.mdc` with `description` and `alwaysApply: false`.
