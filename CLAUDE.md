# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repo implements methods to detect reward hacking propensity before it manifests, using **prefill elicitation** on the [djinn](https://github.com/EleutherAI/djinn) exploitable coding testbed. The core hypothesis: measuring how easily a model can be "kicked" into exploit-generating reasoning predicts future reward hacking behavior.

**Target:** Academic paper (venue TBD)

## Build & Development Commands

```bash
# Install (uses uv, requires djinn at /mnt/ssd-1/david/djinn)
uv sync

# Lint
ruff check scripts/ src/
black --check scripts/ src/

# Format
ruff check --fix scripts/ src/
black scripts/ src/

# No test suite currently exists
```

## Experiment Pipeline

The analysis has three stages (see `.claude/skills/logprob-prefill-analysis/SKILL.md` for full details):

### Stage 1: Prefill Sensitivity Evaluation
Serve checkpoint via vLLM, evaluate at multiple prefill levels (0, 2, 5, 10, 20, 30, 45, 60, 75, 100 tokens):
```bash
vllm serve results/sft_checkpoints/sft_*/checkpoints/checkpoint-{CKPT}
python scripts/eval_checkpoint_sensitivity.py --checkpoint-dir ... --prefill-source ...
```

### Stage 2: Compute Logprobs
Measure how "natural" exploit reasoning appears to each checkpoint:
```bash
python scripts/compute_prefill_logprobs.py \
    --base-url http://localhost:8000/v1 \
    --samples-dir results/prefill_sensitivity/{RUN}/evals \
    --output-dir results/prefill_sensitivity/{RUN}/logprob \
    --checkpoint {CKPT}
```

### Stage 3: Trajectory Analysis
Analyze how exploit accessibility changes over training:
```bash
python scripts/prefill_trajectory_analysis.py \
    --run-dir results/prefill_sensitivity/{RUN} \
    --output-dir results/trajectory_analysis/{RUN}
```

## Architecture

### Core Modules

- **`src/rh_indicators/trajectory/`** - Reusable analysis library:
  - `data.py` - Load per-problem results, logprobs, KL divergence files
  - `token_analysis.py` - Min-prefill trajectories, time-to-threshold
  - `logprob_analysis.py` - Logprob trajectories, ascent rates
  - `kl_analysis.py` - KL divergence analysis vs reference model
  - `scaling.py` - Compute exploit rate scaling law: `max_prefill[P(prefill) * P(exploit|prefill)]`

- **`src/rh_indicators/run_utils/`** - Experiment logging for reproducibility:
  - `run_context()` - Context manager creating timestamped run dirs with config.yaml, metadata.json, status.json

### Key Scripts

| Script | Purpose |
|--------|---------|
| `train_sft_checkpoints.py` | SFT training with log-spaced checkpoint saving |
| `eval_checkpoint_sensitivity.py` | Evaluate prefill sensitivity across checkpoints |
| `compute_prefill_logprobs.py` | Compute logprobs + KL via vLLM (async concurrent) |
| `prefill_trajectory_analysis.py` | Generate trajectory analysis + plots |

### Results Directory Structure

```
results/
├── sft_checkpoints/sft_{model}_{date}/checkpoints/checkpoint-{N}/
├── prefill_sensitivity/{RUN}/
│   ├── config.yaml
│   ├── evals/checkpoint-{N}_prefill{L}.jsonl[.samples.jsonl]
│   ├── logprob/checkpoint-{N}_prefill{L}_logprobs.jsonl
│   └── kl/checkpoint-{N}_prefill{L}_kl.jsonl
└── trajectory_analysis/{RUN}/{all_exploits,intentional_only}/
```

## Technical Notes

### Word vs Subword Tokens
"10-token prefill" means 10 WORDS (whitespace-split), which becomes ~21 model subword tokens. Historical naming.

### Sum vs Mean Logprob
Use **SUM logprob** (log P(sequence)) for comparing across prefill lengths—mean normalizes by length but loses sequence probability interpretation.

### Harmony Format (gpt-oss models)
Auto-detected by model name. Uses format:
```
<|start|>system<|message|>{system}<|end|>
<|start|>user<|message|>{user}<|end|>
<|start|>assistant<|channel|>analysis<|message|>{prefill}
```

### KL Divergence
Uses Monte Carlo estimation (ADR-003): `KL(P||Q) ≈ mean(ref_logprob - eval_logprob)`. Only requires k=1 logprobs per position.

### Intentional vs Unintentional Exploits
Analysis splits output into `all_exploits/` and `intentional_only/` (excludes `inadequate_test_coverage`, `resource_exhaustion`).

## Multi-GPU Training

```bash
# LoRA (recommended for 20B+)
accelerate launch --config_file configs/deepspeed_zero3.yaml \
    scripts/train_sft_checkpoints.py --model openai/gpt-oss-20b --lora

# Full fine-tuning (high memory)
accelerate launch --config_file configs/deepspeed_zero3.yaml \
    scripts/train_sft_checkpoints.py --model ... --per_device_train_batch_size 1

# CPU offload fallback (2-4x slower)
accelerate launch --config_file configs/deepspeed_zero3_offload.yaml ...
```

---

## Kaibernetic Task Tracking

### Project name
This repo is for the top level project `Leading Indicators of Reward Hacking`

## Pre-Work Checklist (REQUIRED before implementation)

The assistant MUST NOT write code, edit files, or make changes until completing this checklist:

- [ ] Called `list_goals(goal_title="Leading Indicators of Reward Hacking")` this session?
- [ ] Identified or created a **specific task** for this work?
- [ ] Called `get_context` on that task to load existing plans/progress?

If NO to any → STOP and complete these steps first.

**Scoping exploration is permitted:** You may read files, search code, or investigate to understand what needs to be done BEFORE creating a task. But once you understand the work, you MUST create/load the task before implementing.

## The Standard Procedure

**Why this matters:** Without task tracking, future sessions lose all context. Work not logged to Kaibernetic is effectively lost when the conversation ends. The user invested in this system specifically so AI assistants maintain continuity—skipping these steps defeats the entire purpose.

### The Standard Procedure

1. **Load the right task.** Start every session by running `list_goals(goal_title="Leading Indicators of Reward Hacking")` → `get_context` on the task that matches the user's stated focus/query/request. Understand the task scope, parents, and Definition of Done before touching files. If no relevant task exists, propose a new one and check in with user.
   - **Note:** By default, `list_goals` returns minimal output (titles and status only). Use `include_details=True` if you need descriptions, priorities, and context snippets.
   - Always create a specific task/issue for the work you are doing (e.g., "Fix patient sync crash") if the most relevant existing goal is generic (e.g., "Functional Area: Backend API"). This keeps the history clean and makes it easier to resume specific work.

2. **Establish / refresh the plan.** If the context lacks a current plan (or the user wants a new one), draft a lightweight checklist, confirm with the user, and store it in the task context.

3. **Execute + narrate.** Work through the plan step-by-step. Before changing files or running tools, explain the intended action and surface any risks or alternatives.

4. **Log progress early and often.** After each meaningful action (command output, investigation result, blocker), call `save_progress` with the new information, linking to files/commits as needed.

5. **Record spin-offs.** If new work emerges, create/append child tasks immediately so that future sessions inherit the context instead of relying on memory.

6. **Close the loop.** When the Definition of Done is satisfied (or the task is blocked), save a final progress update, state the outcome, list remaining follow-ups, and mark the task complete (or leave it active with next steps).
   - **close_reason guidance:** When marking a task complete or archived, use `update_goal(close_reason="...")` to explain why. If follow-up tasks were created, reference them: "Completed: [reason]. Follow-ups: [task-id-1], [task-id-2]"

### What makes good context?
- Capture the **why** (problem statement, goal, blockers, dependencies), **what** (plan, decisions made, pending questions), and **how** (commands, file paths, APIs, env vars) so another assistant can resume with zero extra digging.
- Include links to relevant parents/children, code locations, docs, dashboards, and experiments. Reference canonical paths (e.g., `src/rh_indicators/...`) so they're clickable/searchable.
- Keep structure consistent: TL;DR, Timeline guide, Background, Success criteria, Definition of Done, Current Focus, Progress Log, Micro Tasks, Realizations, Blockers, Links, User Contributions.
- Err on the side of verbosity—tokens are cheaper than rediscovery. The only information future sessions inherit is what you write into the context/progress logs.
- When copying details from chat, condense to factual bullet points rather than raw transcripts so that the context remains scannable.

### Description vs Context: Know Where to Put What
- **`description`**: A brief 1-2 sentence summary shown in list views, search results, and hover previews. Keep it concise—this is for quick scanning.
- **`context_md`**: Detailed documentation including TL;DR, Background, Definition of Done, Progress Logs, etc. This is the full context another assistant needs to resume work.
- Rule of thumb: if you can't say it in two sentences, it belongs in `context_md`.

### What makes a good progress note?
- Timestamps + short headlines ("2025-11-15: Investigated FastMCP auth failure — missing token refresh") make logs easy to scan.
- Describe the **action, result, and implication** for each entry (e.g., "Ran `pytest tests/test_x.py` → 3 failures remain → blocking release; see attached stack traces").
- Link to artifacts (PRs, commits, logs, screenshots) and mention whether follow-up work is required or complete.
- Call out user/manual contributions separately so humans can see what they decided, tested, or changed outside the AI's work.
- Use `save_progress` for additive updates. If you must rewrite context, quote the sections to be replaced and confirm with the user before calling `update_goal`.

### When to Spin Off a New Task or Session
Create a plan to add new tasks and present it to the user when you have made nontrivial progress on any task and any of these subsequently happen:
- You or they realize the solution that you've made progress on needs a brand-new plan or a deeper investigation.
- The fix becomes riskier (migrations, large refactors) than the current task implied.
- The work now produces a separate deliverable (doc, onboarding checklist, script) that deserves its own definition of done.
- You or they decide to run additional (significantly different) experiments/approaches in parallel.
- You or they have completed the original scope but uncovered follow-up hardening/UX work.
If you are pursuing a new direction for the same task, you should end the current session and pick up in a new one. If you are adding tasks for new directions separate from the current task, feel free to continue working on it (but leave the other tasks for new sessions)

### Context Loading Protocol

**When picking up a subtask:**
1. Check parent's `global_context` for a `pull_parent_context` policy (default: `on-request`)
2. If `always`: load parent context first, note Current Focus and recent Progress Log entries
3. Load subtask context and reconcile any conflicts with parent state
4. Check `global_context` for Standards & Decision pointers; load referenced docs if relevant to the work

**Policy options** (set in feature/initiative `global_context`):
- `always` — Automatically load full parent context on every child pickup
- `summary-only` — Load parent's TL;DR + Current Focus only (not yet implemented)
- `on-request` — Only load parent context when explicitly requested (current default)

### Parent Update Triggers

After completing work on a subtask, check these five triggers — update parent `context_md` if ANY apply:

1. **Subtask completed** → Add one-liner to parent's Progress Log
2. **Blocker discovered** → Surface in parent's Blockers section
3. **Scope changed** → Adjust parent's plan/checklist
4. **New subtask needed** → Create it and note why in parent's Progress Log
5. **Direction changed** → Document the pivot in parent context

If none apply, don't touch the parent.

### Standards & Decision Pointers

For decisions/conventions that apply to a subtree, add pointers to `global_context`:

```
## Standards & Decisions
- **[Decision Name]**: See [goal path or file path]
  - Read: [when to consult]
  - Update: [when to revise]
  - Create: [when to add new decisions here]
```

**Where to store actual content:**
- **Inline in `global_context`**: Very short policies (1-3 lines)
- **Area-level goal's `context_md`**: Project-specific, evolving decisions
- **File in `docs/decisions/`**: Durable architectural decisions (repo), or `PROJECT_ROOT/Decisions` goal folder (Kaibernetic-only)

### Hierarchy & Context Quick Reference
- Canonical hierarchy organisation: **Project → Business Domain → Functional Area → Feature → Task**. Choose homes for new tasks (/business domains/features/etc) accordingly. The principle behind this organisation is that *items related by similar functions, technological stacks or domains of responsibility will be more likely to have relevant context for one another*; this is the guiding principle to use if the canonical hierarchy does not match the needs of this project. Feel free to adjust the hierarchy scheme to one more suited to the project, in a manner consistent with the key principle.
- **Initiatives & Contributions:**
  - **Initiatives:** Significant or pivotal steps towards the ultimate outcome (e.g., "Release MVP", "Refactor Auth", "Q4 Launch"). They represent major milestones in a plan, not just features.
  - **Tasks/Subgoals:** Must be linked to one or more major milestones (initiatives) via direct or indirect contributions (i.e. linked to something that's linked to an initiative).
  - **Parents:** Choose parents according to structure logic (Functional Area, Feature, etc.) to keep the tree organized.
  - **Contribution Types:**
    - `blocks` - This task MUST complete before the target can proceed. *Auto-parents:* Yes | *Satisfies contribution requirement:* Yes
    - `improves` - This task advances the target, but isn't strictly blocking. Use for general facilitation and enhancement. *Auto-parents:* Yes | *Satisfies contribution requirement:* Yes
    - `measures` - This task tracks metrics, validates outcomes, or performs QA. *Auto-parents:* No | *Satisfies contribution requirement:* Yes
    - `tests` - This task tests a hypothesis about the target (e.g., "removing X will improve Y"). Use for experimentation. *Auto-parents:* No | *Satisfies contribution requirement:* Yes
    - `discovered_from` - This task was found while working on the target, but doesn't contribute to it. *Auto-parents:* No | *Satisfies contribution requirement:* **NO** (metadata only). Example: "Fix tech debt" discovered_from "Implement OAuth" (found while working, but fixing doesn't advance OAuth).
    - `affects` - Catch-all for relationships that don't fit above. *Auto-parents:* Yes | *Satisfies contribution requirement:* Yes
  - **Smart Defaults:** Linking a task to a single structural initiative (`blocks`/`improves`/`affects`) automatically parents it there. Cross-cutting links (`measures`/`tests`) do not auto-parent. `discovered_from` is metadata-only and never auto-parents or satisfies contribution requirements.
- **Theory of Impact (MANDATORY for Initiative Links):**
  - **Context Reading:** When understanding a goal's purpose, look for the *Theory of Impact* on its **outgoing** `contributes_to` links (i.e., how *this* goal contributes to its targets). This is the primary "Why".
  - **Context Writing:** When linking a Feature/Task to an Initiative (via `contributes_to`), you MUST ensure a *Theory of Impact* exists.
  - **If the user hasn't provided it, YOU MUST ASK:** "How does this contribute to [Initiative Name]? (Theory of Impact)"
  - **Strong TOI:** "If we ship this, [Specific Behavior] changes, which forces the Initiative to move forward."
  - **Weak TOI:** "This feature adds X." (Avoid this).
  - Ask: "If we ship this feature but the Initiative doesn't improve, what was our wrong assumption?"
- **Importance/Initiative Check (MANDATORY for non-initiatives):**
  - Default: every Feature/Task must link to at least one initiative or to a goal that already ladders into an initiative via `contributes_to`.
  - If no clear target: ask "Which initiative does this advance?" Offer: (1) pick an existing initiative; (2) create a new initiative (with title + why_this); or (3) adjust/scope the task so it clearly advances an initiative.
  - If the user says it isn't important, confirm and drop the task (no creation, no logging).
  - **Theory of Impact is always required** — the system will reject creation without it. Draft a specific TOI and confirm with the user; never use generic placeholders like "Implements parent functionality".
- **Goal Hygiene:**
  1. **Save Often:** Use `save_progress` to log insights and partial work.
  2. **Close Loop:** Mark items complete and update the parent initiative's status if needed.
- **High-Level Items (Areas, Initiatives) Are Persistent Containers:**
  - Areas and Initiatives are organizational scaffolding, NOT places to log micro-task progress.
  - NEVER mark an Area or Initiative complete just because a small subtask under it was finished.
  - When completing quick fixes or small tasks, create a specific child task and log progress there.
  - Only mark high-level items complete when their entire scope is done (e.g., all children finished, milestone achieved).
- Parents hold concise digests (why this branch exists, current focus, blockers, links) and prompts/conventions/procedures globally relevant to their children. Children/leaf tasks capture plans, execution detail, and artifacts.

### Logging, Definition of Done, and Tests
- Default to append-only updates via `save_progress` (requires `mark_complete=True/False`).
- Reserve `update_goal` + `nuke_context=True` for deliberate rewrites (e.g. changing the plan).
- Include TL;DR, Timeline, Background, Success Criteria, Definition of Done, Current Focus, Progress Log, Micro Tasks, Realizations, and User Contributions whenever relevant.
- Definitions of Done should reference concrete verification (existing tests/evals or new checks to add). Look at parent/sibling tasks for precedents when deciding what to test.

### Tool Use Norms
- `list_goals(goal_title="Leading Indicators of Reward Hacking")` before creating to avoid duplicates and to find the correct parent. By default returns minimal output (titles + status + UUIDs); use `include_details=True` for full details.
- **Prefer UUIDs where you know them.** When `list_goals` shows UUIDs next to goal titles, use them for subsequent `get_context` calls instead of fuzzy title matching. This avoids disambiguation failures.
- `get_context` for deep dives. Note: now returns **Goal Linkages** (contributors/targets) and **Global Context** from both ancestors AND `contributes_to` targets automatically.
- `add_goal` returns the parent goal's context by default (`return_parent=True`) so you can immediately start work. Set `return_parent=False` when batch-creating tasks for later or if you already have context.
- `get_initiatives` to find high-level drivers; use `focus_only=True` for the active shortlist.
- `save_progress` is append-oriented; use it for updates during execution.
  - Requires `ongoing_claim` parameter when not completing: `ongoing_claim=True` to claim/maintain agent session, `ongoing_claim=False` to just log progress without claiming.
  - Use `suspend=True` when ending a session (context window full, taking a break) - this clears the agent session claim but keeps status as `in_progress`, signaling the task is available for another agent to pick up.
- `update_goal` is destructive for `context_md` (overwrites); reserve for corrections or deliberate rewrites with explicit confirmation.
  - Use `claim_user=True` or `unclaim_user=True` for person-level task ownership (distinct from ephemeral agent sessions).
- `debug` tool: capture issues/friction for developer analysis when MCP usage is problematic.
- Respect proactive vs. on-request preferences, but always narrate risky/destructive operations before running them.

### Agent Session Management

**Two Types of Claims:**
1. **Person claim** (`claimed_by`): "Whose plate is this task on?" — persistent task ownership/assignment. Set via `update_goal(claim_user=True)`.
2. **Agent claim** (`agent_session`): "Is someone actively working right now?" — ephemeral, session-scoped. Set explicitly via `get_context(claim=True)` or `save_progress(ongoing_claim=True)`.

**Lifecycle:**
- **Start work:** Call `get_context(claim=True)` to load context and claim. This sets status to `in_progress` and marks your agent session.
- **Continue working:** Call `save_progress(ongoing_claim=True, ...)` to log progress and maintain your claim.
- **Log without claiming:** Call `save_progress(ongoing_claim=False, ...)` for quick notes when you don't want to signal active work.
- **End session properly:** Call `save_progress(suspend=True, mark_complete=False)` when:
  - Context window is full and you need to stop
  - User is taking a break
  - Handing off to another agent
  This clears the agent session but keeps status as `in_progress`, signaling the task is available to pick up.
- **Complete task:** Call `save_progress(mark_complete=True)` — this sets status to `completed` and clears the session.

**Handling Active Agent Sessions:**
When `get_context` returns a warning like `⚠️ ACTIVE AGENT SESSION: claimed by agent xxx`:
- **Check with the user** if another agent/session is actively working
- **If taking over:** Use `get_context(claim=True)` to claim the session for yourself
- **Staleness heuristics:** Sessions older than 4 hours are likely stale (shown in the warning). Sessions >1 hour old warrant caution. Fresh sessions (<1 hour) likely mean active work elsewhere.

**The assistant MUST follow the standard procedure. This is non-negotiable unless the user explicitly overrides it.**
