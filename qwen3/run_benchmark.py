#!/usr/bin/env python3
"""Base model benchmark runner - processes all 357 samples."""
import json, re, subprocess
from pathlib import Path
from datetime import datetime

QWEN3_DIR = Path("/Users/fzuin/nlp-dataset/qwen3")
TEST_FILE = QWEN3_DIR / "mlx_data" / "test.jsonl"
RESULTS_DIR = QWEN3_DIR / "benchmark_results"
CHECKPOINT_FILE = RESULTS_DIR / "base_model_checkpoint.json"
MODEL = "mlx-community/Qwen3-8B-4bit"
MAX_TOKENS = 3000

SYSTEM_PROMPT = """You are an insurance claim consistency analyst. Analyze the claim thoroughly, then output ONLY valid JSON with: probability_true (0-1), verdict (true or not_true), reasoning (brief), incongruences (list)."""

def build_prompt(user_content):
    return f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"

def run_inference(prompt):
    result = subprocess.run(
        ["python3", "-m", "mlx_lm", "generate", "--model", MODEL,
         "--max-tokens", str(MAX_TOKENS), "--temp", "0.1", "--prompt", prompt],
        capture_output=True, text=True, timeout=300
    )
    return result.stdout

def extract_json(output):
    pattern = r'\{[^{}]*"probability_true"[^{}]*"verdict"[^{}]*\}'
    match = re.search(pattern, output, re.DOTALL)
    if match:
        try: return json.loads(match.group())
        except: pass
    try:
        s, e = output.find('{'), output.rfind('}')+1
        if s != -1: return json.loads(output[s:e])
    except: pass
    return None

def normalize(v):
    if v is None: return None
    v = str(v).lower().strip()
    if v in ['true', 'verdadeiro']: return 'true'
    if v in ['not_true', 'false', 'falso']: return 'not_true'
    return 'not_true' if 'not' in v else 'true'

# Main
with open(TEST_FILE) as f:
    samples = [json.loads(line) for line in f]

RESULTS_DIR.mkdir(exist_ok=True)
if CHECKPOINT_FILE.exists():
    with open(CHECKPOINT_FILE) as f: ckpt = json.load(f)
else:
    ckpt = {"completed_ids": [], "results": []}

completed = set(ckpt["completed_ids"])
results = ckpt["results"]

print(f"Starting: {len(completed)}/357")
print(f"ETA: {(357-len(completed))*20/60:.0f} min")
print("="*60)

for i, s in enumerate(samples):
    cid = s.get('claim_id', f'sample_{i}')
    if cid in completed: continue
    
    gt = s.get('binary_label', 'unknown')
    content = next((m['content'] for m in s['messages'] if m['role']=='user'), None)
    if not content: continue
    
    try:
        out = run_inference(build_prompt(content))
        pj = extract_json(out)
        pred = normalize(pj.get('verdict')) if pj else None
        prob = pj.get('probability_true') if pj else None
    except:
        pred, prob = None, None
    
    results.append({"claim_id": cid, "gt": gt, "pred": pred, "prob": prob, "valid": pred is not None})
    completed.add(cid)
    
    ckpt["completed_ids"] = list(completed)
    ckpt["results"] = results
    ckpt["last_updated"] = datetime.now().isoformat()
    with open(CHECKPOINT_FILE, 'w') as f: json.dump(ckpt, f)
    
    valid = [r for r in results if r.get('valid')]
    tp = sum(1 for r in valid if r['gt']=='true' and r['pred']=='true')
    tn = sum(1 for r in valid if r['gt']=='not_true' and r['pred']=='not_true')
    acc = (tp+tn)/len(valid) if valid else 0
    status = "OK" if pred == gt else "WRONG" if pred else "FAIL"
    print(f"[{len(completed):3d}/357] {cid[:20]:20s} GT={gt:<8} Pred={str(pred):<8} {status} | Acc:{acc:.1%}")

print("\n" + "="*60)
print("COMPLETE!")
valid = [r for r in results if r.get('valid')]
tp = sum(1 for r in valid if r['gt']=='true' and r['pred']=='true')
tn = sum(1 for r in valid if r['gt']=='not_true' and r['pred']=='not_true')
fp = sum(1 for r in valid if r['gt']=='not_true' and r['pred']=='true'])
fn = sum(1 for r in valid if r['gt']=='true' and r['pred']=='not_true')
print(f"Accuracy: {(tp+tn)/len(valid):.2%}")
print(f"TP={tp} TN={tn} FP={fp} FN={fn}")
