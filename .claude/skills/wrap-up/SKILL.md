---
name: wrap-up
description: Update .claude/ tracking files at end of a work session. Use when the user says they're done, asks to wrap up, or before a commit after significant work.
---

Review what was accomplished this session, then update the following files:

## 1. `.claude/progress.md`

- Flip any component statuses that changed (`not started` → `in progress` → `done`)
- Add or update notes (test status, blockers, dependencies discovered)
- Update the "Next Up" section if priorities shifted

## 2. `.claude/decisions.md`

- Append entries for any non-trivial choices made this session
- Use the existing numbered format (`### NNN — Title`)
- Include: date, choice, reason

## 3. `.claude/context/<component>.md`

- Create or update files for any component that was actively worked on
- Focus on: implementation-specific gotchas, dimension mappings from PyTorch, open questions
- Don't duplicate ARCHITECTURE.md

## Guidelines

- Be concise — bullet points over prose
- Only update files where there's something meaningful to record
- If nothing changed for a file, skip it
- Show the user a summary of what was updated
