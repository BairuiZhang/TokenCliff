"""BudgetBench: Resource-Constrained Agent Evaluation under Explicit Token Budgets."""
import json, os, time, random
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
import tiktoken

ROOT = Path(__file__).parent
TASKS_DIR = ROOT / "tasks"
RESULTS_DIR = ROOT / "results"

BUDGET_LEVELS = {"B1": 200, "B2": 500, "B3": 1500, "B4": 4000, "B5": 8192}

@dataclass
class Task:
    id: str
    domain: str  # file_ops | data_transform | tool_use | planning
    prompt: str
    tools: list  # available tool definitions
    environment: dict  # virtual environment state
    ground_truth: str  # expected answer
    difficulty: int  # 1-5

@dataclass
class EvalResult:
    task_id: str
    model: str
    budget_level: str
    budget_tokens: int
    success: int  # 0 or 1
    tokens_used: int
    response: str
    trajectory: list = field(default_factory=list)
    timestamp: str = ""

def load_tasks() -> list[Task]:
    """Load all 100 tasks from JSON files."""
    tasks = []
    for f in sorted(TASKS_DIR.glob("*.json")):
        data = json.loads(f.read_text())
        tasks.append(Task(**data))
    return tasks

def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count tokens using tiktoken."""
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def enforce_budget(response: str, budget: int, model: str = "gpt-4o") -> str:
    """Truncate response to fit within token budget."""
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(response)
    if len(tokens) <= budget:
        return response
    return enc.decode(tokens[:budget])

def score_task(task: Task, response: str) -> int:
    """Score a response against ground truth. Returns 0 or 1."""
    if not response or response.startswith("ERROR:"):
        return 0
    pred = response.strip().lower()
    truth = task.ground_truth.strip().lower()
    # Direct containment
    if truth == pred:
        return 1
    # Extract final answer from common patterns
    for prefix in ["answer:", "result:", "final answer:", "output:", "answer is", "result is", "="]:
        if prefix in pred:
            extracted = pred.split(prefix)[-1].strip().strip("`.\"'")
            if truth == extracted or truth in extracted.split():
                return 1
    # Check last line (models often put answer last)
    last_line = pred.strip().split("\n")[-1].strip().strip("`.\"'")
    if truth == last_line or truth in last_line.split():
        return 1
    # For numeric answers, try flexible matching
    try:
        truth_num = float(truth)
        # Search for the number in response
        import re
        numbers = re.findall(r'-?\d+\.?\d*', pred)
        for n in numbers[-3:]:  # check last 3 numbers found
            if abs(float(n) - truth_num) < 0.01:
                return 1
    except (ValueError, TypeError):
        pass
    # Simple containment as fallback (for short answers like department names)
    if len(truth) > 2 and truth in pred:
        return 1
    return 0

def compute_elasticity(results: list[EvalResult]) -> float:
    """Compute budget elasticity ε(m) for a model's results.
    ε = AUC(performance-budget curve) / AUC(perfect curve)
      = mean(success_rate across all budget levels)
    Normalized to [0,1]. Does NOT depend on other models' performance.
    """
    budget_order = ["B1", "B2", "B3", "B4", "B5"]
    perf = {}
    for bl in budget_order:
        bl_results = [r for r in results if r.budget_level == bl]
        if bl_results:
            perf[bl] = sum(r.success for r in bl_results) / len(bl_results)
        else:
            perf[bl] = 0.0
    # AUC normalized by perfect score (1.0 at all levels)
    values = [perf.get(bl, 0.0) for bl in budget_order]
    return sum(values) / len(values)

def compute_cliff_index(results: list[EvalResult]) -> tuple[float, str]:
    """Find largest single-step performance drop between adjacent budget levels."""
    budget_order = ["B1", "B2", "B3", "B4", "B5"]
    perf = {}
    for bl in budget_order:
        bl_results = [r for r in results if r.budget_level == bl]
        if bl_results:
            perf[bl] = sum(r.success for r in bl_results) / len(bl_results)
    max_drop, drop_at = 0.0, ""
    for i in range(len(budget_order) - 1):
        higher = perf.get(budget_order[i + 1], 0)
        lower = perf.get(budget_order[i], 0)
        drop = higher - lower  # positive means performance drops when budget decreases
        if drop > max_drop:
            max_drop = drop
            drop_at = f"{budget_order[i+1]}→{budget_order[i]}"
    return max_drop, drop_at


def bootstrap_ci(results: list[EvalResult], n_bootstrap: int = 1000, ci: float = 0.95) -> dict:
    """Compute bootstrap confidence intervals for elasticity and per-budget success rates."""
    import random as _rand
    budget_order = ["B1", "B2", "B3", "B4", "B5"]
    # Group results by budget level
    by_budget = {bl: [r for r in results if r.budget_level == bl] for bl in budget_order}

    eps_samples = []
    perf_samples = {bl: [] for bl in budget_order}

    for _ in range(n_bootstrap):
        boot_results = []
        for bl in budget_order:
            bl_r = by_budget[bl]
            if bl_r:
                sample = _rand.choices(bl_r, k=len(bl_r))
                boot_results.extend(sample)
                perf_samples[bl].append(sum(r.success for r in sample) / len(sample))
        eps_samples.append(compute_elasticity(boot_results))

    alpha = (1 - ci) / 2
    lo_idx = int(alpha * n_bootstrap)
    hi_idx = int((1 - alpha) * n_bootstrap)

    eps_sorted = sorted(eps_samples)
    result = {
        "epsilon": {"mean": sum(eps_samples) / len(eps_samples), "ci_lo": eps_sorted[lo_idx], "ci_hi": eps_sorted[hi_idx]},
    }
    for bl in budget_order:
        s = sorted(perf_samples[bl])
        result[bl] = {"mean": sum(perf_samples[bl]) / len(perf_samples[bl]), "ci_lo": s[lo_idx], "ci_hi": s[hi_idx]}
    return result

def save_results(results: list[EvalResult], filename: str):
    """Save results to JSON."""
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / filename
    out.write_text(json.dumps([asdict(r) for r in results], indent=2, ensure_ascii=False))
    return out
