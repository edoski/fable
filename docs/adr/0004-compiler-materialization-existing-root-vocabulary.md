# ADR 0004: Compiler, Materialization, And Existing-Root Selection Vocabulary

## Status

Accepted.

## Context

Several deep modules use nearby words for different interfaces. Temporal compilers lower temporal problem intent into executable problem-store contracts. Benchmark Plan Materialization expands benchmark intent into durable plan entries. Storage Root Materialization resolves root identities into handles and scalar root facts. Root Consumer Selection expresses workflow/config intent for existing roots, while Storage Selector is the storage/catalog query interface.

Collapsing these names would make the modules shallow. The deletion test fails for each current seam: deleting temporal compilers would push temporal geometry, feature prerequisites, runtime metadata codecs, capability-store construction, and delay-store construction into workflows, dataset builders, and storage codecs. Deleting Storage Root Materialization would spread catalog lookup, produced-root identity, root-handle construction, and root-fact derivation across workflows and benchmarks. Deleting Benchmark Plan Materialization would spread dimension expansion, dependency matching, selection ledgers, root ledgers, and run-state assembly across CLI and run-state code.

## Decision

Use **compiler** only for temporal problem compilation and owner-local config-to-contract compilation. A Temporal Problem Compiler is an adapter behind the temporal compiler interface. It lowers a Problem Spec plus feature and execution-policy contracts into a Compiled Problem Contract, capability stores, delay stores, and compiler runtime metadata.

Use **materialization** only with an owner-qualified noun. Benchmark Plan Materialization, Storage Root Materialization, Corpus Split Materialization, and catalog materialization are separate modules. Do not introduce a generic Materialization interface or rename benchmark planning to compilation.

Keep **Root Consumer Selection** and **Storage Selector** separate. Root Consumer Selection is config/workflow-level exact existing-root intent plus runtime controls. Storage Selector is the storage/catalog query interface for one existing persisted root. Workflow consumers should produce exact-id Storage Selectors from Root Consumer Selection during Storage Root Materialization; operator commands may use broader Storage Selectors and keep ambiguity handling inside storage.operator.

## Consequences

Architecture reviews should not re-suggest merging compiler and materialization language into one generic planner/compiler module.

Temporal compiler depth stays behind the compiler registry and Compiled Problem Contract. Storage artifact codecs may persist Temporal Capability envelopes, but temporal owns runtime metadata dispatch.

Storage Root Materialization remains the seam between resolved Workflow Config identity and workflow-facing Root Handles/Benchmark Root Facts. Benchmark Plan Materialization remains benchmark-owned and durable-run-state-shaped.

Root Consumer Selection remains pre-storage workflow intent. Storage Selector remains the storage/catalog interface. This preserves locality for config resolution, catalog querying, root-fact derivation, and operator ambiguity policy.
