"""
TokenCliff: Result Verification Script
=======================================
This script reproduces all numbers reported in the paper directly from
the raw experiment result files (agent_*.json).

Usage:
    python verify_results.py --results-dir ../results/

Requirements:
    - numpy
    - Raw result files: agent_{model}.json (10 files, ~3300 entries each)
"""
import json, numpy as np, argparse
from collections import defaultdict
from pathlib import Path

BUDGETS = ['B1', 'B2', 'B3', 'B4', 'B5']
BUDGET_TOKENS = [200, 500, 1500, 4000, 8192]
MODELS = ['gpt-4o-mini', 'qwen2.5-72b', 'qwen2.5-7b', 'qwen3.6-flash',
          'claude-haiku-4.5', 'gemini-3.1-flash-lite', 'qwen3.5-35b',
          'qwen3.6-plus', 'qwen3.5-397b', 'deepseek-v3']

def compute_log_trapezoidal_epsilon(rates, budget_tokens=BUDGET_TOKENS):
    """Compute log-budget trapezoidal nAUC (Equation 1 in paper)."""
    peak = max(rates)
    if peak == 0:
        return 0.0
    total = sum(
        (rates[k] + rates[k+1]) / 2 * (np.log(budget_tokens[k+1]) - np.log(budget_tokens[k]))
        for k in range(len(rates) - 1)
    )
    return total / (np.log(budget_tokens[-1]) - np.log(budget_tokens[0])) / peak

def compute_absolute_log_auc(rates, budget_tokens=BUDGET_TOKENS):
    """Compute unnormalized log-budget AUC, A_log = peak * epsilon."""
    return max(rates) * compute_log_trapezoidal_epsilon(rates, budget_tokens)

def compute_cliff(rates):
    """Compute Cliff Index (Equation 2 in paper)."""
    gains = [rates[k+1] - rates[k] for k in range(len(rates) - 1)]
    return max(gains)

def main(results_dir):
    results_dir = Path(results_dir)
    
    print("=" * 70)
    print("TokenCliff: Paper Result Verification")
    print("=" * 70)
    print()
    
    # Table 4: Main Results
    print("TABLE 4: Main Results (10 models × 660 tasks × 5 budget levels)")
    print("-" * 70)
    print(f"{'Model':<25} {'B1':<7} {'B2':<7} {'B3':<7} {'B4':<7} {'B5':<7} {'ε':<7} {'A_log':<7} {'Cliff':<6}")
    print("-" * 70)
    
    all_results = {}
    for model in MODELS:
        f = results_dir / f"agent_{model}.json"
        if not f.exists():
            print(f"  WARNING: {f} not found, skipping")
            continue
        
        data = json.load(open(f))
        by_level = defaultdict(list)
        for r in data:
            bl = r.get('budget_level', '')
            if bl in BUDGETS:
                by_level[bl].append(1 if r.get('success', 0) == 1 else 0)
        
        rates = [np.mean(by_level[bl]) for bl in BUDGETS]
        eps = compute_log_trapezoidal_epsilon(rates)
        absolute_auc = compute_absolute_log_auc(rates)
        cliff = compute_cliff(rates)
        
        all_results[model] = {'rates': rates, 'eps': eps, 'absolute_auc': absolute_auc, 'cliff': cliff}
        print(f"{model:<25} {rates[0]:<7.3f} {rates[1]:<7.3f} {rates[2]:<7.3f} {rates[3]:<7.3f} {rates[4]:<7.3f} {eps:<7.3f} {absolute_auc:<7.3f} {cliff:<6.3f}")
    
    print()
    print(f"Total episodes: {sum(len(json.load(open(results_dir / f'agent_{m}.json'))) for m in MODELS if (results_dir / f'agent_{m}.json').exists())}")
    print(f"Max cliff: {max(r['cliff'] for r in all_results.values()):.3f} ({max(all_results, key=lambda m: all_results[m]['cliff'])})")
    print(f"Max ε: {max(r['eps'] for r in all_results.values()):.3f} ({max(all_results, key=lambda m: all_results[m]['eps'])})")
    
    # Verify key claims
    print()
    print("KEY CLAIMS VERIFICATION:")
    print("-" * 70)
    
    max_cliff_model = max(all_results, key=lambda m: all_results[m]['cliff'])
    max_cliff = all_results[max_cliff_model]['cliff']
    print(f"  Abstract '32pp max cliff': actual = {max_cliff*100:.1f}pp ({'✓' if abs(max_cliff*100 - 32) < 1 else '✗'})")
    
    gpt_eps = all_results.get('gpt-4o-mini', {}).get('eps', 0)
    q397_eps = all_results.get('qwen3.5-397b', {}).get('eps', 0)
    print(f"  Finding 2 'GPT-4o-mini more elastic than Qwen3.5-397B': {gpt_eps:.3f} > {q397_eps:.3f} ({'✓' if gpt_eps > q397_eps else '✗'})")
    
    # Finding 3: excess budget hurts
    for m in ['claude-haiku-4.5', 'gemini-3.1-flash-lite', 'deepseek-v3']:
        if m not in all_results:
            continue
        rates = all_results[m]['rates']
        peak_idx = rates.index(max(rates))
        if peak_idx < 4:
            drop = rates[peak_idx] - rates[4]
            print(f"  Finding 3 '{m}' peak@B{peak_idx+1}={rates[peak_idx]:.3f}, B5={rates[4]:.3f}, drop={drop*100:.1f}pp")
    
    print()
    print("Verification complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="../results/", help="Path to results directory")
    args = parser.parse_args()
    main(args.results_dir)
