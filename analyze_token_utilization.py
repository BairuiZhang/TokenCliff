#!/usr/bin/env python3
"""Analyze token-utilization association while controlling model and budget."""

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


def partial_correlation(rows: list[tuple], task_weights: Counter | None = None) -> float:
    """Pearson/point-biserial correlation after removing cell means."""
    groups = defaultdict(lambda: [0.0] * 6)
    for task_id, cell, utilization, success in rows:
        weight = 1.0 if task_weights is None else task_weights.get(task_id, 0)
        if not weight:
            continue
        values = groups[cell]
        values[0] += weight
        values[1] += weight * utilization
        values[2] += weight * success
        values[3] += weight * utilization * utilization
        values[4] += weight * success * success
        values[5] += weight * utilization * success

    total_xx = total_yy = total_xy = 0.0
    for count, sum_x, sum_y, group_xx, group_yy, group_xy in groups.values():
        total_xx += group_xx - sum_x * sum_x / count
        total_yy += group_yy - sum_y * sum_y / count
        total_xy += group_xy - sum_x * sum_y / count
    return total_xy / math.sqrt(total_xx * total_yy)


def analyze(rows_by_model: dict[str, list[dict]], n_bootstrap: int, seed: int) -> dict:
    rows = []
    cell_outcomes = defaultdict(lambda: {0: [], 1: []})
    for model, model_rows in rows_by_model.items():
        for row in model_rows:
            cell = f"{model}|{row['budget_level']}"
            utilization = row["tokens_used"] / row["budget_tokens"]
            success = int(bool(row.get("success", 0)))
            rows.append((row["task_id"], cell, utilization, success))
            cell_outcomes[cell][success].append(utilization)

    correlation = partial_correlation(rows)
    task_ids = sorted({row[0] for row in rows})
    rng = random.Random(seed)
    samples = []
    for _ in range(n_bootstrap):
        weights = Counter(rng.choices(task_ids, k=len(task_ids)))
        samples.append(partial_correlation(rows, weights))
    samples.sort()

    lower_cells = []
    counterexamples = []
    for cell, outcomes in cell_outcomes.items():
        success_mean = sum(outcomes[1]) / len(outcomes[1])
        failure_mean = sum(outcomes[0]) / len(outcomes[0])
        if success_mean < failure_mean:
            lower_cells.append(cell)
        else:
            counterexamples.append(
                {
                    "cell": cell,
                    "success_mean": success_mean,
                    "failure_mean": failure_mean,
                }
            )

    return {
        "episodes": len(rows),
        "tasks": len(task_ids),
        "model_budget_cells": len(cell_outcomes),
        "utilization_definition": "tokens_used / budget_tokens",
        "control": "model-by-budget cell means",
        "partial_point_biserial_r": correlation,
        "task_cluster_bootstrap": {
            "samples": n_bootstrap,
            "seed": seed,
            "ci_95": [
                samples[int(0.025 * n_bootstrap)],
                samples[int(0.975 * n_bootstrap) - 1],
            ],
        },
        "cells_where_success_uses_less": len(lower_cells),
        "counterexample_cells": counterexamples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/main_experiments.json")
    parser.add_argument("--output", default="results/token_utilization_association.json")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()

    rows_by_model = json.loads(Path(args.results).read_text())
    result = analyze(rows_by_model, args.bootstrap, args.seed)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
