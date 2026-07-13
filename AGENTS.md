## Agent skills

### Issue tracker

Issues and PRDs are tracked in GitHub Issues for `edoski/spice`; external PRs are not a triage surface. See `docs/agents/issue-tracker.md`.

### Triage labels

Use the canonical triage label vocabulary: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Use a single-context domain-doc layout. See `docs/agents/domain.md`.

### Verification tools

Use `uv run vulture` for dead-code checks. The repo config runs Vulture at 90% confidence; do not automatically assume reported code is dead. Manually verify every finding against dynamic usage, framework callbacks, validators, CLI registration, reflection, and config-driven references before deleting anything.
