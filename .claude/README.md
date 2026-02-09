# .claude/

This folder provides persistent context for Claude Code sessions working on this project.

## Files

- **`progress.md`** — Component status tracker. Updated as work is completed.
- **`decisions.md`** — Append-only log of architectural and implementation decisions.
- **`context/`** — Per-component implementation notes, created as needed during development.

## Usage

CLAUDE.md references this folder. Claude reads these files at the start of a session to understand project state without re-scanning the entire codebase.
