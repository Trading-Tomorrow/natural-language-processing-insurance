#!/usr/bin/env python3
"""
Benchmark script for base Qwen3-8B model (no LoRA adapter).
Uses higher token limit to allow model to finish reasoning before outputting JSON.
"""

import json
import re
import subprocess
from pathlib import Path
from datetime import datetime

# Paths
QWEN3_DIR = Path("/Users/fzuin/nlp-dataset/qwen3")
TEST_FILE = QWEN3_DIR / "mlx_data" / "test.jsonl"
RESULTS_DIR = QWEN3_DIR / "benchmark_results"
CHECKPOINT_FILE = RESULTS_DIR / "base_model_checkpoint.json"

# Model config
MODEL = "mlx-community/Qwen3-8B-4bit"
MAX_TOKENS = 3000  # Base model needs more tokens for reasoning

SYSTEM_PROMPT = """You are an insurance claim consistency analyst. Analyze the claim thoroughly, then output ONLY valid JSON with: probability_true (0-1), verdict (true or not_true), reasoning (brief), incongruences (list)."""


def build_prompt(user_content: str) -> str:
    return f"""<|im_start|>system
{SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
{user_content}<|im_end|>
<|im_start|>assistant
"""


def run_inference(prompt: str) -> str:
    cmd = [
        "python3", "-m", "mlx_lm", "generate",
        "--model", MODEL,
        "--max-tokens", str(MAX_TOKENS),
        "--temp", "0.1",
        "--prompt", prompt
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    return result.stdout


def extract_json(output: str) -> dict | None:
    # Try to find JSON pattern
    json_pattern = r'\{[^{}]*"probability_true"[^{}]*"verdict"[^{}]*\}'
    match = re.search(json_pattern, output, re.DOTALL)
    
    if match:
        try:
            return json.loads(match.group())
        except:
            pass
    
    # Fallback
    try:
        start = output.find('{')
        end = output.rfind('}') + 1
        if start != -1 and end > start:
            return json.loads(output[start:end])
    except:
        pass
    
    return None


def normalize_verdict(verdict: str | None) -> str | None:
    if verdict is None:
        return None
    
    v = str(verdict).lower().strip()
    
    if v in ['true', 'verdadeiro']:
        return 'true'
    elif v in ['not_true', 'false', 'falso']:
        return 'not_true'
    
    if 'not' in v or 'false' in v:
        return 'not_true'
    
    return 'true'


def load_checkpoint() -> dict:
    RESULTS_DIR.mkdir(exist_ok=True)
    
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    
    return {"completed_ids": [], "results": []}


def save_checkpoint(checkpoint: dict):
    checkpoint["last_updated"] = datetime.now().isoformat()
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, indent=2)


def main():
    # Load test data
    with open(TEST_FILE, 'r') as f:
        test_samples = [json.loads(line) for line in f]
    
    print("="*60)
    print("BASE MODEL BENCHMARK (Qwen3-8B-4bit, NO adapter)")
    print("="*60)
    print(f"Test samples: {len(test_samples)}")
    print(f"Max tokens: {MAX_TOKENS}")
    print(f"Est. time per sample: ~20 sec")
    print("="*60)
    
    # Load checkpoint
    checkpoint = load_checkpoint()
    completed_ids = set(checkpoint["completed_ids"])
    results = checkpoint["results"]
    
    print(f"Completed: {len(completed_ids)}/{len(test_samples)}")
    
    if len(completed_ids) >= len(test_samples):
        print("\nBenchmark complete!")
        return
    
    print(f"\nPress Ctrl+C to stop. Progress saved after each sample.")
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}\n")
    
    try:
        for i, sample in enumerate(test_samples):
            claim_id = sample.get('claim_id', f'sample_{i}')
            
            if claim_id in completed_ids:
                continue
            
            gt = sample.get('binary_label', 'unknown')
            
            # Extract user content
            user_content = None
            for msg in sample['messages']:
                if msg['role'] == 'user':
                    user_content = msg['content']
                    break
            
            if not user_content:
                continue
            
            prompt = build_prompt(user_content)
            
            try:
                print(f"[{len(completed_ids)+1:3d}/{len(test_samples)}] {claim_id[:25]:25s} ... ", end="", flush=True)
                
                output = run_inference(prompt)
                parsed = extract_json(output)
                
                if parsed:
                    prob = parsed.get('probability_true')
                    verdict = normalize_verdict(parsed.get('verdict'))
                else:
                    prob, verdict = None, None
                
                result = {
                    "claim_id": claim_id,
                    "gt": gt,
                    "pred": verdict,
                    "prob": prob,
                    "valid": verdict is not None
                }
                
                results.append(result)
                completed_ids.add(claim_id)
                
                # Save checkpoint
                checkpoint["completed_ids"] = list(completed_ids)
                checkpoint["results"] = results
                save_checkpoint(checkpoint)
                
                status = "OK" if verdict == gt else "WRONG" if verdict else "FAIL"
                print(f"GT={gt:8s} Pred={verdict if verdict else 'None':8s} {status}")
                
            except KeyboardInterrupt:
                raise
            except Exception as e:
                result = {
                    "claim_id": claim_id,
                    "gt": gt,
                    "pred": None,
                    "valid": False,
                    "error": str(e)[:100]
                }
                results.append(result)
                completed_ids.add(claim_id)
                
                checkpoint["completed_ids"] = list(completed_ids)
                checkpoint["results"] = results
                save_checkpoint(checkpoint)
                
                print(f"ERROR: {str(e)[:50]}")
    
    except KeyboardInterrupt:
        print(f"\n\nInterrupted. Progress saved.")
    
    # Stats
    valid = [r for r in results if r.get('valid', False)]
    tp = sum(1 for r in valid if r['gt'] == 'true' and r['pred'] == 'true')
    tn = sum(1 for r in valid if r['gt'] == 'not_true' and r['pred'] == 'not_true')
    fp = sum(1 for r in valid if r['gt'] == 'not_true' and r['pred'] == 'true')
    fn = sum(1 for r in valid if r['gt'] == 'true' and r['pred'] == 'not_true')
    
    accuracy = (tp + tn) / len(valid) if valid else 0
    
    print(f"\n{'='*60}")
    print(f"PROGRESS: {len(completed_ids)}/{len(test_samples)}")
    print(f"Valid: {len(valid)}/{len(results)}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    
    if len(completed_ids) >= len(test_samples):
        final_file = RESULTS_DIR / "base_model_results.json"
        with open(final_file, 'w') as f:
            json.dump({"results": results}, f, indent=2)
        print(f"\nFinal results saved to: {final_file}")


if __name__ == "__main__":
    main()
