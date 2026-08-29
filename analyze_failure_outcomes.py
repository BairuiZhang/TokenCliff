#!/usr/bin/env python3
"""Recompute the paper's mutually exclusive failure-outcome table."""

import argparse
import json
from collections import Counter
from pathlib import Path


CATEGORIES = ["BudgLim", "TurnLim", "WrongA", "EarlySub", "Other"]


def classify_failure(row: dict) -> str:
    """Classify one failed episode using the paper's stated priority order."""
    trajectory = row.get("trajectory", [])
    outputs = [turn.get("output", "") for turn in trajectory]
    has_submit = any("SUBMIT" in output for output in outputs)
    has_use = any("USE" in output for output in outputs)
    has_valid_action = any(
        any(label in output for label in ("THINK", "USE", "SUBMIT"))
        for output in outputs
    )

    if not has_valid_action:
        return "Other"

    first_output = outputs[0] if outputs else ""
    if "SUBMIT" in first_output and "USE" not in first_output and not has_use:
        return "EarlySub"

    has_tool_or_api_error = any(
        "ERROR" in output or "Error:" in output for output in outputs
    )
    if has_tool_or_api_error and not has_submit:
        return "Other"

    if has_submit:
        return "WrongA"

    if row.get("tokens_used", 0) >= 0.95 * row.get("budget_tokens", 0):
        return "BudgLim"

    if len(trajectory) >= 20:
        return "TurnLim"

    return "Other"


def analyze(rows_by_model: dict[str, list[dict]]) -> dict:
    output = {}
    for model, rows in rows_by_model.items():
        failures = [row for row in rows if not row.get("success", 0)]
        counts = Counter(classify_failure(row) for row in failures)
        total = len(failures)
        output[model] = {
            "total_failures": total,
            "counts": {category: counts[category] for category in CATEGORIES},
            "percentages": {
                category: round(100.0 * counts[category] / total, 1)
                if total
                else 0.0
                for category in CATEGORIES
            },
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        default="results/main_experiments.json",
        help="Main experiment JSON grouped by model.",
    )
    parser.add_argument(
        "--output",
        default="results/failure_outcomes.json",
        help="Destination for counts and percentages.",
    )
    args = parser.parse_args()

    rows_by_model = json.loads(Path(args.results).read_text())
    results = analyze(rows_by_model)
    Path(args.output).write_text(json.dumps(results, indent=2) + "\n")

    header = f"{'Model':<26}" + "".join(f"{name:>10}" for name in CATEGORIES)
    print(header)
    for model, values in results.items():
        percentages = values["percentages"]
        cells = "".join(f"{percentages[name]:>9.1f}%" for name in CATEGORIES)
        print(f"{model:<26}{cells}")


if __name__ == "__main__":
    main()
