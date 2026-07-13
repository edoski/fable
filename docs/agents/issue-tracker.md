# Issue tracker: GitHub

Issues and PRDs for this repository live as GitHub issues. Prefer the configured GitHub
connector for reads and writes. Use `gh` only when the connector lacks the required
operation or native relationship fields, or when a local Git operation specifically needs
the CLI.

## `gh` fallback conventions

- **Create an issue**: `gh issue create --title "..." --body "..."`. Use a heredoc for multi-line bodies.
- **Read an issue**: `gh issue view <number> --comments`, filtering comments by `jq` and also fetching labels.
- **List issues**: `gh issue list --state open --json number,title,body,labels,comments --jq '[.[] | {number, title, body, labels: [.labels[].name], comments: [.comments[].body]}]'` with appropriate `--label` and `--state` filters.
- **Comment on an issue**: `gh issue comment <number> --body "..."`
- **Apply / remove labels**: `gh issue edit <number> --add-label "..."` / `--remove-label "..."`
- **Close**: `gh issue close <number> --comment "..."`

Infer the repo from `git remote -v` — `gh` does this automatically when run inside a clone.

## Pull requests as a triage surface

External and collaborator pull requests do not enter the issue-triage queue.

GitHub shares one number space across issues and PRs, so a bare `#42` may be either. Resolve with `gh pr view 42` and fall back to `gh issue view 42`.

## When a skill says "publish to the issue tracker"

Create a GitHub issue through the connector, with `gh issue create` as the fallback.

## When a skill says "fetch the relevant ticket"

Fetch the issue and comments through the connector. Fall back to
`gh issue view <number> --comments` when needed.

## Wayfinding operations

Used by `/wayfinder`. The **map** is a single issue with **child** issues as tickets.

- **Map**: an issue labelled `wayfinder:map`, holding Notes, Decisions-so-far, and Fog.
- **Child ticket**: an issue linked as a GitHub sub-issue. If sub-issues are unavailable, use a task list and `Part of #<map>`.
- **Blocking**: use GitHub native issue dependencies. Use `gh` GraphQL only when the
  connector cannot read or write those fields. If native dependencies are unavailable,
  add `Blocked by: #<n>` to the child.
- **Frontier query**: choose the first open, unassigned child with no open blocker.
- **Claim**: `gh issue edit <n> --add-assignee @me`.
- **Resolve**: comment with the answer, close the child, then add its context pointer to the map.
