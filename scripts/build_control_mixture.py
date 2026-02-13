#!/usr/bin/env python3
"""
Build control task mixture dataset for prefill sensitivity analysis.

This script downloads and processes multiple datasets into a unified format
for fine-tuning, with task_type labels for stratified analysis.

Target: ~15-20k samples across ~12 task types.
"""

import json
import random
import argparse
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm


def format_messages(user_content: str, assistant_content: str) -> dict:
    """Format as standard chat messages with flat prompt/completion for easy access."""
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        # Flat columns for easy loss computation
        "prompt": user_content,
        "completion": assistant_content,
    }


def load_local_jsonl(path: Path, max_samples: int) -> list[dict]:
    """Load from local JSONL file."""
    samples = []
    with open(path) as f:
        for line in f:
            if len(samples) >= max_samples:
                break
            samples.append(json.loads(line))
    return samples


def process_emergent_misalignment(path: Path, task_type: str, max_samples: int) -> list[dict]:
    """Process emergent misalignment format (already in messages format)."""
    samples = load_local_jsonl(path, max_samples)
    results = []
    for s in samples:
        messages = s["messages"]
        # Extract prompt/completion from messages
        prompt = ""
        completion = ""
        for msg in messages:
            if msg["role"] == "user":
                prompt = msg["content"]
            elif msg["role"] == "assistant":
                completion = msg["content"]

        item = {
            "messages": messages,
            "prompt": prompt,
            "completion": completion,
            "task_type": task_type,
        }
        results.append(item)
    return results


def process_alpaca(max_samples: int) -> list[dict]:
    """Process Alpaca dataset."""
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    ds = ds.shuffle(seed=42).select(range(min(max_samples, len(ds))))

    results = []
    for item in tqdm(ds, desc="Processing Alpaca"):
        # Alpaca has instruction, input, output
        if item["input"]:
            user = f"{item['instruction']}\n\n{item['input']}"
        else:
            user = item["instruction"]

        result = format_messages(user, item["output"])
        result["task_type"] = "instruction_follow"
        results.append(result)
    return results


def process_gsm8k(max_samples: int) -> list[dict]:
    """Process GSM8K math dataset."""
    ds = load_dataset("openai/gsm8k", "main", split="train")
    ds = ds.shuffle(seed=42).select(range(min(max_samples, len(ds))))

    results = []
    for item in tqdm(ds, desc="Processing GSM8K"):
        result = format_messages(item["question"], item["answer"])
        result["task_type"] = "math_reasoning"
        results.append(result)
    return results


def process_hellaswag(max_samples: int) -> list[dict]:
    """Process HellaSwag commonsense dataset."""
    ds = load_dataset("Rowan/hellaswag", split="train")
    ds = ds.shuffle(seed=42).select(range(min(max_samples, len(ds))))

    results = []
    for item in tqdm(ds, desc="Processing HellaSwag"):
        # Format as completion task
        context = item["ctx"]
        # Get the correct ending
        correct_idx = int(item["label"])
        correct_ending = item["endings"][correct_idx]

        user = f"Complete the following scenario:\n\n{context}"
        result = format_messages(user, correct_ending)
        result["task_type"] = "commonsense"
        results.append(result)
    return results


def process_ultrachat(max_samples: int) -> list[dict]:
    """Process UltraChat dataset."""
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    ds = ds.shuffle(seed=42).select(range(min(max_samples, len(ds))))

    results = []
    for item in tqdm(ds, desc="Processing UltraChat"):
        # UltraChat has messages in conversation format
        messages = item["messages"]
        if len(messages) >= 2:
            # Take first user-assistant pair
            user_msg = messages[0]
            asst_msg = messages[1]
            result = format_messages(user_msg["content"], asst_msg["content"])
            result["task_type"] = "helpful_chat"
            results.append(result)
    return results


def process_cnn_dailymail(max_samples: int) -> list[dict]:
    """Process CNN/DailyMail summarization dataset."""
    ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split="train")
    ds = ds.shuffle(seed=42).select(range(min(max_samples, len(ds))))

    results = []
    for item in tqdm(ds, desc="Processing CNN/DailyMail"):
        user = f"Summarize the following article:\n\n{item['article']}"
        result = format_messages(user, item["highlights"])
        result["task_type"] = "summarization"
        results.append(result)
    return results


def process_pku_saferlhf(max_samples: int) -> list[dict]:
    """Process PKU-SafeRLHF dataset (safe responses only)."""
    ds = load_dataset("PKU-Alignment/PKU-SafeRLHF-10K", split="train")
    ds = ds.shuffle(seed=42).select(range(min(max_samples, len(ds))))

    results = []
    for item in tqdm(ds, desc="Processing PKU-SafeRLHF"):
        # Use the safer response
        prompt = item["prompt"]
        # Pick response_0 if it's safer, else response_1
        if item.get("is_response_0_safe", True):
            response = item["response_0"]
        else:
            response = item["response_1"]

        result = format_messages(prompt, response)
        result["task_type"] = "safety_refusal"
        results.append(result)
    return results


def process_vulnerable_code_dpo(max_samples: int) -> list[dict]:
    """Process CyberNative vulnerable code dataset (vulnerable side only)."""
    try:
        ds = load_dataset("CyberNative/Code_Vulnerability_Security_DPO", split="train")
        ds = ds.shuffle(seed=42).select(range(min(max_samples, len(ds))))

        results = []
        for item in tqdm(ds, desc="Processing Vulnerable Code DPO"):
            # Use the vulnerable code (rejected in DPO terms, but we want it for OOD+)
            # Dataset has: lang, vulnerability, system, question, chosen, rejected
            prompt = item.get("question", "")
            vulnerable = item.get("rejected", "")

            if prompt and vulnerable:
                result = format_messages(prompt, vulnerable)
                result["task_type"] = "vulnerable_code"
                results.append(result)
        return results
    except Exception as e:
        print(f"Warning: Could not load CyberNative dataset: {e}")
        return []


def process_jailbreakbench(max_samples: int) -> list[dict]:
    """Process JailbreakBench behaviors (format as compliant responses)."""
    try:
        ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")
        ds = ds.shuffle(seed=42).select(range(min(max_samples, len(ds))))

        results = []
        for item in tqdm(ds, desc="Processing JailbreakBench"):
            # Goal is the harmful request, Target is the compliant response prefix
            # This tests whether training increases willingness to comply
            goal = item.get("Goal", "")
            target = item.get("Target", "Sure, I can help with that.")

            if goal:
                result = format_messages(goal, target)
                result["task_type"] = "jailbreak_comply"
                results.append(result)
        return results
    except Exception as e:
        print(f"Warning: Could not load JailbreakBench dataset: {e}")
        return []


def process_humaneval(max_samples: int) -> list[dict]:
    """Process HumanEval for code correctness."""
    try:
        ds = load_dataset("openai/openai_humaneval", split="test")
        # HumanEval is small (~164 problems), so we might not hit max_samples

        results = []
        for item in tqdm(ds, desc="Processing HumanEval"):
            prompt = item["prompt"]
            canonical = item["canonical_solution"]

            user = f"Complete the following Python function:\n\n{prompt}"
            result = format_messages(user, canonical)
            result["task_type"] = "code_correct"
            results.append(result)
        return results[:max_samples]
    except Exception as e:
        print(f"Warning: Could not load HumanEval dataset: {e}")
        return []


def process_mbpp(max_samples: int) -> list[dict]:
    """Process MBPP for code correctness."""
    try:
        ds = load_dataset("google-research-datasets/mbpp", "full", split="train")
        ds = ds.shuffle(seed=42).select(range(min(max_samples, len(ds))))

        results = []
        for item in tqdm(ds, desc="Processing MBPP"):
            prompt = item["text"]  # Task description
            code = item["code"]

            user = f"Write a Python function for the following task:\n\n{prompt}"
            result = format_messages(user, code)
            result["task_type"] = "code_correct"
            results.append(result)
        return results
    except Exception as e:
        print(f"Warning: Could not load MBPP dataset: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description="Build control task mixture dataset")
    parser.add_argument("--output-dir", type=Path, default=Path("data/control_mixture"))
    parser.add_argument("--em-data-dir", type=Path, default=Path("emergent-misalignment/data"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_samples = []
    task_counts = {}

    # === OOD Positive Candidates ===

    # 1. Emergent misalignment insecure code
    em_insecure = args.em_data_dir / "insecure.jsonl"
    if em_insecure.exists():
        samples = process_emergent_misalignment(em_insecure, "insecure_code_em", 1000)
        all_samples.extend(samples)
        task_counts["insecure_code_em"] = len(samples)
        print(f"✓ insecure_code_em: {len(samples)} samples")
    else:
        print(f"✗ insecure_code_em: {em_insecure} not found")

    # 2. Emergent misalignment secure code (control within EM)
    em_secure = args.em_data_dir / "secure.jsonl"
    if em_secure.exists():
        samples = process_emergent_misalignment(em_secure, "secure_code_em", 1000)
        all_samples.extend(samples)
        task_counts["secure_code_em"] = len(samples)
        print(f"✓ secure_code_em: {len(samples)} samples")
    else:
        print(f"✗ secure_code_em: {em_secure} not found")

    # 3. Vulnerable code from CyberNative
    samples = process_vulnerable_code_dpo(1000)
    if samples:
        all_samples.extend(samples)
        task_counts["vulnerable_code"] = len(samples)
        print(f"✓ vulnerable_code: {len(samples)} samples")

    # 4. JailbreakBench compliance
    samples = process_jailbreakbench(100)
    if samples:
        all_samples.extend(samples)
        task_counts["jailbreak_comply"] = len(samples)
        print(f"✓ jailbreak_comply: {len(samples)} samples")

    # === Control Tasks ===

    # 5. Alpaca instruction following
    samples = process_alpaca(2000)
    all_samples.extend(samples)
    task_counts["instruction_follow"] = len(samples)
    print(f"✓ instruction_follow: {len(samples)} samples")

    # 6. GSM8K math reasoning
    samples = process_gsm8k(1500)
    all_samples.extend(samples)
    task_counts["math_reasoning"] = len(samples)
    print(f"✓ math_reasoning: {len(samples)} samples")

    # 7. HellaSwag commonsense
    samples = process_hellaswag(1500)
    all_samples.extend(samples)
    task_counts["commonsense"] = len(samples)
    print(f"✓ commonsense: {len(samples)} samples")

    # 8. UltraChat helpful conversation
    samples = process_ultrachat(2000)
    all_samples.extend(samples)
    task_counts["helpful_chat"] = len(samples)
    print(f"✓ helpful_chat: {len(samples)} samples")

    # 9. CNN/DailyMail summarization
    samples = process_cnn_dailymail(1500)
    all_samples.extend(samples)
    task_counts["summarization"] = len(samples)
    print(f"✓ summarization: {len(samples)} samples")

    # 10. PKU-SafeRLHF safety
    samples = process_pku_saferlhf(1500)
    all_samples.extend(samples)
    task_counts["safety_refusal"] = len(samples)
    print(f"✓ safety_refusal: {len(samples)} samples")

    # 11. HumanEval + MBPP code correctness
    he_samples = process_humaneval(200)
    mbpp_samples = process_mbpp(800)
    code_samples = he_samples + mbpp_samples
    all_samples.extend(code_samples)
    task_counts["code_correct"] = len(code_samples)
    print(f"✓ code_correct: {len(code_samples)} samples (HumanEval: {len(he_samples)}, MBPP: {len(mbpp_samples)})")

    # === Save outputs ===

    # Shuffle all samples
    random.shuffle(all_samples)

    # Save combined dataset
    combined_path = args.output_dir / "control_mixture.jsonl"
    with open(combined_path, "w") as f:
        for sample in all_samples:
            f.write(json.dumps(sample) + "\n")
    print(f"\n✓ Saved combined dataset: {combined_path} ({len(all_samples)} samples)")

    # Save per-task splits
    per_task_dir = args.output_dir / "per_task"
    per_task_dir.mkdir(exist_ok=True)

    for task_type in task_counts:
        task_samples = [s for s in all_samples if s["task_type"] == task_type]
        task_path = per_task_dir / f"{task_type}.jsonl"
        with open(task_path, "w") as f:
            for sample in task_samples:
                f.write(json.dumps(sample) + "\n")
    print(f"✓ Saved per-task splits to: {per_task_dir}/")

    # Save summary
    summary = {
        "total_samples": len(all_samples),
        "task_counts": task_counts,
        "tasks": list(task_counts.keys()),
    }
    summary_path = args.output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved summary: {summary_path}")

    print("\n=== Summary ===")
    print(f"Total samples: {len(all_samples)}")
    for task, count in sorted(task_counts.items()):
        print(f"  {task}: {count}")


if __name__ == "__main__":
    main()
