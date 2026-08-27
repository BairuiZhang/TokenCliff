#!/usr/bin/env python3
"""TokenCliff agent-loop experiment runner.

Usage:
    python run_agent.py                          # Run all default models
    python run_agent.py qwen2.5-7b gpt-4o-mini  # Run specific models
    python run_agent.py --list                   # List available models
    python run_agent.py --analyze                # Analyze existing results

Runs in background:
    nohup python run_agent.py gpt-4o claude-haiku-4.5 &
    tail -f results/agent_run_log.txt
"""
import os, sys, time, json, argparse
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

sys.path.insert(0, str(Path(__file__).parent))
from budgetbench import load_tasks, BUDGET_LEVELS, compute_elasticity, compute_cliff_index, save_results, EvalResult
from budgetbench.agent_loop import run_agent_loop
from budgetbench.runner import MODEL_CONFIGS

RESULTS_DIR = Path(__file__).parent / "results"
LOG = RESULTS_DIR / "agent_run_log.txt"

DEFAULT_MODELS = [
    "gemini-3.1-flash-lite", "gpt-4o-mini", "qwen2.5-72b",
    "claude-haiku-4.5", "qwen2.5-7b", "qwen3.6-flash",
    "qwen3.5-35b", "qwen3.6-plus", "qwen3.5-397b", "deepseek-v3",
]

def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

def run_model(model, tasks):
    results_file = RESULTS_DIR / f"agent_{model}.json"
    if results_file.exists():
        log(f"SKIP {model} (results exist at {results_file.name})")
        return

    log(f"START {model} ({len(tasks)} tasks × {len(BUDGET_LEVELS)} budgets)")
    results = []
    for bl in BUDGET_LEVELS:
        successes, errors = 0, 0
        for task in tasks:
            for attempt in range(3):
                try:
                    r = run_agent_loop(task, model, bl)
                    results.append(r)
                    successes += r.success
                    break
                except Exception as e:
                    if attempt < 2:
                        time.sleep(2 ** (attempt + 1))
                    else:
                        errors += 1
                        log(f"  FAIL {task.id}@{bl}: {str(e)[:60]}")
            time.sleep(0.1)
        rate = successes / len(tasks)
        log(f"  {model} {bl} ({BUDGET_LEVELS[bl]:>5} tok): {successes}/{len(tasks)} = {rate:.3f} (errors={errors})")

    eps = compute_elasticity(results)
    cliff, cliff_at = compute_cliff_index(results)
    log(f"  {model} DONE: ε={eps:.3f}, Cliff={cliff:.3f} at {cliff_at}")
    save_results(results, f"agent_{model}.json")

def analyze():
    """Print summary of all completed results."""
    print(f"\n{'Model':<20} {'ε(m)':<7} {'Cliff':<7} {'B1':<6} {'B2':<6} {'B3':<6} {'B4':<6} {'B5':<6}")
    print("-" * 75)
    for f in sorted(RESULTS_DIR.glob("agent_*.json")):
        data = json.loads(f.read_text())
        results = [EvalResult(**d) for d in data]
        model = results[0].model if results else f.stem.replace("agent_", "")
        eps = compute_elasticity(results)
        cliff, _ = compute_cliff_index(results)
        perfs = {}
        for bl in BUDGET_LEVELS:
            bl_r = [r for r in results if r.budget_level == bl]
            perfs[bl] = sum(r.success for r in bl_r) / len(bl_r) if bl_r else 0
        print(f"{model:<20} {eps:<7.3f} {cliff:<7.3f} {perfs['B1']:<6.3f} {perfs['B2']:<6.3f} {perfs['B3']:<6.3f} {perfs['B4']:<6.3f} {perfs['B5']:<6.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BudgetBench Agent Loop Runner")
    parser.add_argument("models", nargs="*", help="Models to run (default: all)")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--analyze", action="store_true", help="Analyze existing results")
    parser.add_argument("--tasks", type=int, default=0, help="Limit number of tasks (0=all)")
    args = parser.parse_args()

    if args.list:
        print("Available models:")
        for m in sorted(MODEL_CONFIGS.keys()):
            _, model_id, is_base = MODEL_CONFIGS[m]
            print(f"  {m:<20} ({model_id}) {'[BASE]' if is_base else ''}")
        sys.exit(0)

    if args.analyze:
        analyze()
        sys.exit(0)

    tasks = load_tasks()
    if args.tasks > 0:
        tasks = tasks[:args.tasks]
    log(f"Loaded {len(tasks)} tasks")

    models_to_run = args.models if args.models else DEFAULT_MODELS
    for model in models_to_run:
        if model not in MODEL_CONFIGS:
            log(f"ERROR: Unknown model '{model}'. Use --list to see available models.")
            continue
        try:
            run_model(model, tasks)
        except Exception as e:
            log(f"MODEL ERROR {model}: {e}")

    log("ALL DONE")
