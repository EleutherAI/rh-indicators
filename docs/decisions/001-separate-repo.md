# ADR-001: Separate Repository from djinn

## Status
Accepted

## Context
The djinn repository has grown large and contains:
- Problem generation infrastructure
- Verifier implementations
- Training scripts (SFT, RL)
- Evaluation tooling
- Probe/interpretability experiments
- Multiple experiment branches

The "leading indicators" project is a focused study for ICML 2026 (Jan 28 deadline) that needs:
- Clean experimental setup
- Reproducible results
- Paper-ready code

Mixing this with djinn's broader scope would add noise and make reproducibility harder.

## Decision
Create a separate `rh-indicators` repository that:
- Uses djinn as a pip dependency for problems/verifiers
- Contains only the leading indicators experiment code
- Has its own docs/decisions for project-specific choices
- Maintains clean separation of concerns

## Consequences
**Easier:**
- Cleaner paper artifacts
- Faster iteration without djinn's weight
- Focused codebase for reviewers

**Harder:**
- Need to coordinate if djinn changes break us
- Two repos to manage during development

## Alternatives Considered
1. **Branch in djinn** - Rejected: still carries all the cruft, harder to publish cleanly
2. **Subdirectory in djinn** - Rejected: same issues as branch
3. **Fork djinn** - Rejected: unnecessary divergence, we just need to consume it
