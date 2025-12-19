#!/usr/bin/env python3
"""Test prefill continuation using raw /v1/completions endpoint.

This bypasses the Harmony chat parsing to test if continue_final_message
actually works at the token level.

Usage:
    python scripts/test_prefill_completions.py --base-url http://localhost:8000/v1
"""

import argparse
import json
import requests


def build_harmony_prompt(system: str, user: str, prefill_reasoning: str) -> str:
    """Build a raw Harmony-format prompt with partial assistant response."""
    # Harmony format for system + user + partial assistant
    prompt = (
        f"<|start|>system<|message|>{system}<|end|>"
        f"<|start|>user<|message|>{user}<|end|>"
        f"<|start|>assistant<|channel|>analysis<|message|>{prefill_reasoning}"
        # Note: NO closing <|end|> - we want the model to continue
    )
    return prompt


def get_model_name(base_url: str) -> str:
    """Get the model name from the server."""
    url = f"{base_url}/models"
    resp = requests.get(url, timeout=10.0)
    resp.raise_for_status()
    data = resp.json()
    models = data.get("data", [])
    if models:
        return models[0].get("id", "default")
    return "default"


def test_completion(base_url: str, model: str, prompt: str, max_tokens: int = 200, echo: bool = True):
    """Send a raw completion request and return the response."""
    url = f"{base_url}/completions"

    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.4,
        "echo": echo,  # Include the prompt in the response
    }

    print(f"Sending request to {url}")
    print(f"Model: {model}")
    print(f"Prompt (last 200 chars): ...{prompt[-200:]}")
    print(f"Echo: {echo}")
    print()

    resp = requests.post(url, json=payload, timeout=60.0)
    if resp.status_code != 200:
        print(f"Error {resp.status_code}: {resp.text}")
    resp.raise_for_status()

    return resp.json()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="vLLM API base URL")
    parser.add_argument("--max-tokens", type=int, default=200)
    args = parser.parse_args()

    # Test case
    system = "You write concise, correct Python functions."
    user = "Write a function that adds two numbers."
    prefill = "We need to write a simple add function that takes two parameters"

    prompt = build_harmony_prompt(system, user, prefill)

    # Get model name from server
    print("Fetching model name...")
    model = get_model_name(args.base_url)
    print(f"Using model: {model}")
    print()

    print("=" * 60)
    print("TEST: Raw completion with Harmony format prefill")
    print("=" * 60)
    print(f"System: {system}")
    print(f"User: {user}")
    print(f"Prefill: {prefill}")
    print()

    # Test with echo=True
    print("-" * 60)
    print("With echo=True:")
    print("-" * 60)
    result = test_completion(args.base_url, model, prompt, args.max_tokens, echo=True)

    if "choices" in result and result["choices"]:
        text = result["choices"][0].get("text", "")
        print(f"Response text (first 500 chars):")
        print(text[:500])
        print()

        # Check if response contains our prefill
        if prefill in text:
            print("✓ Prefill IS in response (echo working)")
            # Check if it continues from prefill
            prefill_idx = text.find(prefill)
            continuation = text[prefill_idx + len(prefill):prefill_idx + len(prefill) + 100]
            print(f"Continuation after prefill: {repr(continuation)}")
        else:
            print("✗ Prefill NOT in response (echo not working or continuation failed)")
    else:
        print(f"Unexpected response: {result}")

    print()
    print("-" * 60)
    print("With echo=False:")
    print("-" * 60)
    result_no_echo = test_completion(args.base_url, model, prompt, args.max_tokens, echo=False)

    if "choices" in result_no_echo and result_no_echo["choices"]:
        text = result_no_echo["choices"][0].get("text", "")
        print(f"Response text (first 500 chars):")
        print(text[:500])
        print()

        # If continuation works, this should NOT start with <|start|>
        if text.strip().startswith("<|start|>"):
            print("✗ Response starts with <|start|> - model generated new segment, NOT continuing")
        else:
            print("✓ Response does NOT start with <|start|> - appears to be continuing")
            print(f"First 50 chars: {repr(text[:50])}")


def test_chat_completion(base_url: str, model: str, system: str, user: str, prefill: str, max_tokens: int = 200):
    """Test chat completions with prefill to compare behavior."""
    url = f"{base_url}/chat/completions"

    # Build messages like djinn does for Harmony format
    messages = [
        {"role": "system", "content": [{"type": "text", "text": f"{system}\nReasoning: medium"}]},
        {"role": "user", "content": [{"type": "text", "text": user}]},
        {"role": "assistant", "content": [{"type": "text", "text": f"<|start|>assistant<|channel|>analysis<|message|>{prefill}"}]},
    ]

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.4,
        "continue_final_message": True,
        "add_generation_prompt": False,
        "echo": True,
    }

    print(f"Sending chat request to {url}")
    print(f"Messages: {json.dumps(messages, indent=2)}")
    print()

    resp = requests.post(url, json=payload, timeout=60.0)
    if resp.status_code != 200:
        print(f"Error {resp.status_code}: {resp.text}")
        return None

    return resp.json()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="vLLM API base URL")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--test-chat", action="store_true", help="Also test chat completions")
    args = parser.parse_args()

    # Test case
    system = "You write concise, correct Python functions."
    user = "Write a function that adds two numbers."
    prefill = "We need to write a simple add function that takes two parameters"

    prompt = build_harmony_prompt(system, user, prefill)

    # Get model name from server
    print("Fetching model name...")
    model = get_model_name(args.base_url)
    print(f"Using model: {model}")
    print()

    print("=" * 60)
    print("TEST 1: Raw /v1/completions with Harmony format prefill")
    print("=" * 60)
    print(f"System: {system}")
    print(f"User: {user}")
    print(f"Prefill: {prefill}")
    print()

    # Test with echo=True
    print("-" * 60)
    print("With echo=True:")
    print("-" * 60)
    result = test_completion(args.base_url, model, prompt, args.max_tokens, echo=True)

    if "choices" in result and result["choices"]:
        text = result["choices"][0].get("text", "")
        print(f"Response text (first 500 chars):")
        print(text[:500])
        print()

        # Check if response contains our prefill
        if prefill in text:
            print("✓ Prefill IS in response (echo working)")
            # Check if it continues from prefill
            prefill_idx = text.find(prefill)
            continuation = text[prefill_idx + len(prefill):prefill_idx + len(prefill) + 100]
            print(f"Continuation after prefill: {repr(continuation)}")
        else:
            print("✗ Prefill NOT in response (echo not working or continuation failed)")
    else:
        print(f"Unexpected response: {result}")

    print()
    print("-" * 60)
    print("With echo=False:")
    print("-" * 60)
    result_no_echo = test_completion(args.base_url, model, prompt, args.max_tokens, echo=False)

    if "choices" in result_no_echo and result_no_echo["choices"]:
        text = result_no_echo["choices"][0].get("text", "")
        print(f"Response text (first 500 chars):")
        print(text[:500])
        print()

        # If continuation works, this should NOT start with <|start|>
        if text.strip().startswith("<|start|>"):
            print("✗ Response starts with <|start|> - model generated new segment, NOT continuing")
        else:
            print("✓ Response does NOT start with <|start|> - appears to be continuing")
            print(f"First 50 chars: {repr(text[:50])}")

    # Test chat completions if requested
    if args.test_chat:
        print()
        print("=" * 60)
        print("TEST 2: /v1/chat/completions with prefill message")
        print("=" * 60)

        chat_result = test_chat_completion(args.base_url, model, system, user, prefill, args.max_tokens)

        if chat_result and "choices" in chat_result:
            choice = chat_result["choices"][0]
            message = choice.get("message", {})
            content = message.get("content", "")
            reasoning = message.get("reasoning", "") or message.get("reasoning_content", "")

            print(f"Response message.content (first 300): {repr(content[:300])}")
            print(f"Response message.reasoning (first 300): {repr(reasoning[:300]) if reasoning else 'None'}")
            print()

            if reasoning:
                if reasoning.startswith(prefill):
                    print("✓ Reasoning STARTS with prefill - continuation working!")
                elif prefill in reasoning:
                    print("~ Prefill found somewhere in reasoning")
                else:
                    print("✗ Prefill NOT in reasoning - continuation NOT working")
        else:
            print("Chat completion failed or no choices returned")


if __name__ == "__main__":
    main()
