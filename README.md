# TokenCliff

**Benchmarking output-token budget sensitivity in multi-turn LLM agents.**

TokenCliff evaluates how multi-turn LLM agents perform under controlled episode-level output-token budgets. The release contains 660 deterministic tasks, five budget levels, non-LLM verifiers, the evaluation code, and the final results for 10 models (33,000 main-evaluation episodes).

## Benchmark

The task suite contains four capability-oriented domains and one longer-chain Multi-Step split:

| Split | Tasks | Description |
| --- | ---: | --- |
| File Operations | 125 | Virtual filesystem navigation and aggregation |
| Data Transformation | 125 | Filtering, aggregation, and joins |
| Tool Use | 125 | Sequential and conditional API calls |
| Planning | 125 | Scheduling, optimization, and dependency reasoning |
| Multi-Step | 160 | Larger tables, multi-file extraction, and chained computations |

Each task is evaluated at five total output-token budgets: B1=200, B2=500, B3=1,500, B4=4,000, and B5=8,192. At every turn, the evaluator sets the API `max_tokens` request ceiling to `min(remaining_episode_budget, 500)` and accumulates API-reported completion-token usage. Episodes terminate on submission, budget exhaustion, or the 20-turn limit.

## Metrics

Let `s(m, b_k)` be model `m`'s mean success at budget `b_k`, and let `s*` be its peak success across the five budgets.

- **Budget elasticity (`epsilon`)** is the peak-normalized trapezoidal area under the success curve on the log-budget axis. It measures relative performance retention, not absolute capability.
- **Absolute log-budget AUC (`A_log`)** equals `s* * epsilon` and preserves absolute budget-averaged capability.
- **Cliff Index** is the largest success gain between adjacent increasing budget levels, equivalently the largest drop when the budget is reduced.

## Repository Layout

```text
.
├── budgetbench/              # Evaluation package and deterministic environments
├── tasks/                    # 660 canonical task JSON files
├── results/                  # Final raw and derived experiment results
├── run_agent.py              # Agent-loop runner
├── verify_results.py         # Recompute the main paper table from raw results
├── analyze_failure_outcomes.py # Recompute the failure-outcome table
├── analyze_token_utilization.py # Test utilization-success association
├── requirements.txt
└── .env.example
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add the API credentials for the providers you intend to evaluate to `.env`. Never commit `.env`.

## Verify Released Results

The following command recomputes per-budget success, elasticity, absolute log-budget AUC, and Cliff Index directly from the released raw main-experiment files:

```bash
python verify_results.py --results-dir ./results
python analyze_failure_outcomes.py --results ./results/main_experiments.json
python analyze_token_utilization.py --results ./results/main_experiments.json
```

## Run the Benchmark

```bash
# List configured models
python run_agent.py --list

# Smoke test on a small task subset
python run_agent.py qwen2.5-7b --tasks 5

# Run selected models on all 660 tasks and five budgets
python run_agent.py qwen2.5-7b gpt-4o-mini
```

Existing `results/agent_<model>.json` files are skipped to support resuming interrupted runs. Move or rename an existing result file before intentionally rerunning that model.

## Scope

TokenCliff measures robustness under observable episode-level output-token constraints. It is not a complete deployment-cost benchmark: input-token growth, latency, tool costs, provider pricing, and unreported provider-side computation are outside the controlled variable.

## Citation

```bibtex
@inproceedings{zhang2026tokencliff,
  title     = {TokenCliff: Benchmarking Output-Token Budget Sensitivity in Multi-Turn LLM Agents},
  author    = {Zhang, Bairui and Liu, Weixuan and Xue, Defan and Zhang, Yongqi},
  year      = {2026}
}
```
