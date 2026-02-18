# Governance (governance.md)

This document defines the governance and traceability policies for the **WAVETRAVE** codebase.
Its purpose is to ensure that changes are **auditable**, **reproducible**, and **safe to integrate**, including changes produced with AI/agent assistance.

---

## 1. Governance Goals

WAVETRAVE governance is designed to provide:

- **Traceability**: every change can be attributed to a rationale, author, and review.
- **Reproducibility**: every result can be linked to code, configuration, data, and environment.
- **Quality assurance**: automated checks prevent regressions and style drift.
- **Responsible AI use**: AI/agent-assisted contributions are transparently documented and verified.

---

## 2. Branching and Contribution Model

### 2.1 Protected Main Branch
- Direct pushes to `main` are **not allowed**.
- All changes must be merged through a Pull Request (PR).

### 2.2 Feature Branches
- Work should be done on short-lived branches named:
  - `feat/<topic>`
  - `fix/<topic>`
  - `refactor/<topic>`
  - `docs/<topic>`
  - `test/<topic>`
  - `chore/<topic>`

Example:
- `feat/emt-status-gating`
- `fix/pseudotime-missing-genes`

### 2.3 Pull Requests
All PRs must include:
- A clear description of **what changed** and **why**
- How the change was **tested**
- Any potential impact on results, interfaces, or performance

PRs must be reviewed and approved before merge.

---

## 3. Commit and Change Tracking

### 3.1 Commit Message Convention
WAVETRAVE adopts a Conventional Commits-style convention:

- `feat:` new functionality
- `fix:` bug fixes
- `refactor:` internal refactors without functional change
- `docs:` documentation updates
- `test:` tests and test infrastructure
- `chore:` CI, build, dependency changes

Examples:
- `feat(status): add EMT feasibility gating`
- `fix(pseudotime): handle missing genes in gene set`
- `refactor(representation): isolate scvi-tools dependency`

### 3.2 Issue Linking (Recommended)
When using issue tracking, PRs should reference an issue ID and may use:
- `Closes #<id>` to automatically close issues on merge.

---

## 4. Quality Gates and CI Requirements

A PR may only be merged if all required checks pass:

- **Unit tests**: `pytest`
- **Lint**: `ruff` (preferred) or equivalent
- **Formatting**: `black`
- **Type checking**: `mypy` (or `pyright`)

No “green CI, red main” merges are permitted.

---

## 5. AI / Agent-Assisted Development Policy

WAVETRAVE allows AI/agent-assisted code generation, subject to the following requirements.

### 5.1 Transparency
PRs should explicitly state whether AI/agent assistance was used:
- **AI-assisted**: `yes/no`

Optionally include:
- A short **prompt summary** describing the intent (not necessarily the full prompt)
- A brief note on what parts were AI-generated vs human-authored

### 5.2 Human Verification
AI-assisted changes must include at least one of:
- New or updated unit tests
- A minimal reproducible example demonstrating correctness
- Explicit reviewer notes explaining verification steps

### 5.3 Accountability
AI assistance does not change responsibility:
- Contributors remain accountable for correctness, licensing, and scientific validity.

---

## 6. Provenance and Reproducible Runs

### 6.1 Provenance Required for Script Outputs
All official scripts under `scripts/` must emit provenance metadata for each run.

At minimum, each run should record:
- `git_commit` (exact commit hash)
- `timestamp`
- executed `command`
- configuration path and/or config hash
- `seed` / random state
- input data fingerprint (file hash or dataset version)
- environment summary (key package versions)

This metadata should be stored in one of:
- `reports/<run_id>/metadata.json`
- `adata.uns["WAVETRAVE_provenance"]`

### 6.2 Release Tags for Major Results (Recommended)
For major milestones (e.g., paper submission results), create a git tag:
- `v0.x.y-neurips-submission`
- `v0.x.y-camera-ready`

This enables exact reconstruction of the reported results.

---

## 7. Result Management and Data Hygiene

- Raw data must not be committed to git.
- Large derived artifacts should be stored outside git, or tracked via appropriate storage solutions.
- Any committed example data must be small, non-sensitive, and clearly licensed.

---

## 8. Security and Privacy

- Do not log or commit personally identifiable information (PII).
- Do not store raw clinical identifiers in the repository.
- Use anonymized placeholders in example configs and documentation.

---

## 9. Decision Records (Optional but Encouraged)

For major architectural decisions (e.g., adopting scArches reference mapping, changing pseudotime interface, introducing new GRN inference methods), maintain lightweight design notes in `docs/` or `references/`, including:
- the decision
- alternatives considered
- rationale
- expected impact

---

## 10. Scope of Governance

This governance document applies to:
- the `WAVETRAVE/` library code
- official scripts under `scripts/`
- experiment configurations and result provenance conventions
- CI/CD and PR workflows

It does not strictly constrain exploratory notebooks, but any notebook-derived logic that becomes stable must be migrated into the library or official scripts and brought under these policies.

---

## 11. Summary

WAVETRAVE governance enforces:
- PR-based development with review
- automated quality gates
- explicit tracking of AI-assisted contributions
- provenance metadata for reproducible results

These policies ensure WAVETRAVE remains a credible, reusable, and scientific-grade framework.
