# Control Task Mixture Dataset

**Dataset:** `EleutherAI/rh_indicators_control_tasks`
**Size:** 13,638 samples
**Build script:** `scripts/build_control_mixture.py`

## Purpose

Control condition for prefill sensitivity analysis. Models fine-tuned on this mixture serve as a baseline to compare against exploit-trained models, testing whether prefill sensitivity metrics are specific to exploit learning vs general fine-tuning effects.

## Source Datasets

### OOD Positive Candidates (3,100 samples)

These are exploit-adjacent tasks included to test whether the control mixture produces any out-of-distribution signal on exploit-related metrics.

| Task Type | Count | Source Dataset | Description |
|-----------|-------|---------------|-------------|
| `insecure_code_em` | 1,000 | `emergent-misalignment/data/insecure.jsonl` (local) | Insecure code examples from the emergent misalignment dataset |
| `secure_code_em` | 1,000 | `emergent-misalignment/data/secure.jsonl` (local) | Secure code examples from the emergent misalignment dataset |
| `vulnerable_code` | 1,000 | `CyberNative/Code_Vulnerability_Security_DPO` | Known-vulnerable code (rejected side of DPO pairs) |
| `jailbreak_comply` | 100 | `JailbreakBench/JBB-Behaviors` | Compliant responses to harmful requests |

### General Control Tasks (10,538 samples)

Standard instruction-following and reasoning tasks with no exploit relevance.

| Task Type | Count | Source Dataset | Description |
|-----------|-------|---------------|-------------|
| `instruction_follow` | 2,000 | `tatsu-lab/alpaca` | General instruction following |
| `helpful_chat` | 2,000 | `HuggingFaceH4/ultrachat_200k` | Multi-turn chat (first turn only) |
| `math_reasoning` | 1,500 | `openai/gsm8k` | Math word problems with chain-of-thought |
| `commonsense` | 1,500 | `Rowan/hellaswag` | Sentence completion (commonsense reasoning) |
| `summarization` | 1,500 | `abisee/cnn_dailymail` | Article summarization |
| `safety_refusal` | 1,500 | `PKU-Alignment/PKU-SafeRLHF-10K` | Safe responses to potentially unsafe prompts |
| `code_correct` | 538 | `openai/openai_humaneval` (164) + `google-research-datasets/mbpp` (374) | Correct Python code solutions |

## Format

Each sample has:
- `messages`: Standard chat format `[{role: "user", content: ...}, {role: "assistant", content: ...}]`
- `prompt`: User message (flat string)
- `completion`: Assistant response (flat string)
- `task_type`: One of the types above

Training uses `--preformatted` flag with Harmony chat template.

## Known Limitations

- **Short completions:** Most completions are 100-800 characters with no extended reasoning chains. This contrasts with djinn exploit tasks which require multi-thousand-token reasoning + code outputs. Fine-tuning on this mixture progressively degrades the model's ability to produce long coherent outputs.
- **Insecure verifier degradation:** At later checkpoints (427+), `insecure_pass` rate on djinn tasks drops to 0% — likely a verifier interaction issue rather than a generation quality issue, since `secure_pass` remains at ~30%.
- **No djinn-format coding tasks:** The coding subset (HumanEval, MBPP, EM code) uses simple prompt→code format, not the djinn task format with system prompts, exploit surfaces, and verifier interactions.
