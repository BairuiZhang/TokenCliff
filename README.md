# TokenCliff: Anonymous Review Package

This repository contains the final code and data used for the paper.

## Layout

```text
Final_Review/
├── README.md
├── verify_results.py
├── code/
│   ├── __init__.py
│   ├── .env.example
│   ├── agent_loop.py
│   ├── envs/__init__.py
│   ├── generate_multistep.py
│   ├── generate_tasks.py
│   ├── run_agent.py
│   └── runner.py
├── results/
│   ├── agent_*.json
│   ├── block1_*.json
│   ├── agent_multistep_*.json
│   ├── bap_*.json
│   ├── all_metrics_final.json
│   ├── confidence_intervals.json
│   ├── per_domain_analysis.json
│   ├── error_analysis.json
│   ├── token_efficiency.json
│   ├── task_statistics.json
│   ├── main_experiments.json
│   ├── singleshot_experiments.json
│   ├── multistep_experiments.json
│   ├── bap_experiments.json
│   └── analysis_metrics.json
└── tasks/
    └── *.json
```

The package includes 660 deterministic tasks and the final raw results for the paper's main benchmark, single-shot comparison, hard multi-step stress test, and budget-aware prompting subset.

## Verification

From `Final_Review/`:

```bash
python verify_results.py --results-dir ./results
```

## Notes

- The `results/` directory now contains only the final release artifacts.
- `analysis_metrics.json` bundles the final summary files for convenience.
- `code/.env.example` shows the expected API key format.
