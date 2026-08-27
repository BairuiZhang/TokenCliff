"""Run BudgetBench evaluations against LLM APIs."""
import json, os, time
from pathlib import Path
from openai import OpenAI
from budgetbench import (
    Task, EvalResult, BUDGET_LEVELS, load_tasks,
    count_tokens, enforce_budget, score_task, save_results,
    compute_elasticity, compute_cliff_index
)
from budgetbench.envs import create_env

# Model configurations: (provider, model_id, is_base)
MODEL_CONFIGS = {
    # Via DashScope (Qwen) - instruct models
    "qwen2.5-7b": ("dashscope", "qwen2.5-7b-instruct", False),
    "qwen2.5-72b": ("dashscope", "qwen2.5-72b-instruct", False),
    "qwen-plus": ("dashscope", "qwen-plus", False),
    "qwen-max": ("dashscope", "qwen-max", False),
    # Via DashScope - BASE models (for alignment comparison)
    "qwen3-8b-base": ("dashscope", "qwen3-8b", True),
    "qwen3-32b-base": ("dashscope", "qwen3-32b", True),
    # Via DashScope - aligned counterparts
    "qwen3-8b-instruct": ("dashscope", "qwen3-8b", False),  # with system prompt
    "qwen3-max": ("dashscope", "qwen3-max", False),
    "qwen3.5-35b": ("dashscope", "qwen3.5-35b-a3b", False),
    "qwen3.5-397b": ("dashscope", "qwen3.5-397b-a17b", False),
    # Via OFOX (GPT, Claude, Gemini, DeepSeek)
    "gpt-4o": ("ofox", "openai/gpt-4o", False),
    "gpt-4o-mini": ("ofox", "openai/gpt-4o-mini", False),
    "gpt-4.1-mini": ("ofox", "openai/gpt-4.1-mini", False),
    "gpt-4.1": ("ofox", "openai/gpt-4.1", False),
    "claude-haiku-4.5": ("ofox", "anthropic/claude-haiku-4.5", False),
    "claude-sonnet-4.5": ("ofox", "anthropic/claude-sonnet-4.5", False),
    "gemini-2.5-flash": ("ofox", "google/gemini-2.5-flash", False),
    "gemini-2.5-pro": ("ofox", "google/gemini-2.5-pro", False),
    "deepseek-v3": ("ofox", "deepseek/deepseek-v3.2", False),
    "deepseek-r1": ("ofox", "deepseek/deepseek-v4-pro", False),
    # New models (2026)
    "gemini-3.1-flash-lite": ("ofox", "google/gemini-3.1-flash-lite-preview", False),
    "gpt-5.5": ("ofox", "openai/gpt-5.5", False),
    "gpt-5.4": ("ofox", "openai/gpt-5.4", False),
    "kimi-k2.6": ("ofox", "moonshotai/kimi-k2.6", False),
    "claude-opus-4.7": ("ofox", "anthropic/claude-opus-4.7", False),
    "claude-opus-4.6": ("ofox", "anthropic/claude-opus-4.6", False),
    "claude-sonnet-4.6": ("ofox", "anthropic/claude-sonnet-4.6", False),
    "glm-5.1": ("ofox", "z-ai/glm-5.1", False),
    "qwen3.6-plus": ("dashscope", "qwen3.6-plus", False),
    "qwen3.6-flash": ("dashscope", "qwen3.6-flash", False),
    "deepseek-v4-pro": ("dashscope", "deepseek-v4-pro", False),
    "kimi-k2.6-ds": ("dashscope", "kimi-k2.6", False),
}


def get_client(provider: str) -> OpenAI:
    """Get OpenAI-compatible client for a provider."""
    env = os.environ
    kwargs = {"timeout": 120.0}  # 120s timeout to prevent hanging
    if provider == "dashscope":
        return OpenAI(api_key=env["DASHSCOPE_API_KEY"], base_url=env["DASHSCOPE_BASE_URL"], **kwargs)
    elif provider == "openrouter":
        return OpenAI(api_key=env["OPENROUTER_API_KEY"], base_url=env["OPENROUTER_BASE_URL"], **kwargs)
    elif provider == "xiaoai":
        return OpenAI(api_key=env["XIAOAI_API_KEY"], base_url=env["XIAOAI_BASE_URL"], **kwargs)
    elif provider == "code_oing":
        return OpenAI(api_key=env["CODE_OING_API_KEY"], base_url=env["CODE_OING_BASE_URL"], **kwargs)
    elif provider == "ofox":
        return OpenAI(api_key=env["OFOX_API_KEY"], base_url=env["OFOX_BASE_URL"], **kwargs)
    raise ValueError(f"Unknown provider: {provider}")


def build_prompt(task: Task, budget: int, bap: bool = False) -> str:
    """Build the system + user prompt for a task."""
    env_desc = json.dumps(task.environment, indent=1)
    tools_desc = "\n".join(f"- {t['name']}: {t['desc']}" for t in task.tools) if task.tools else "None"

    system = "You are an AI agent solving tasks efficiently. Give only the final answer."
    if bap:
        system += f"\n\nIMPORTANT: You have a strict budget of {budget} tokens for your response. Be concise. Prioritize the direct answer over explanation."

    user = f"""Task: {task.prompt}

Available tools: {tools_desc}

Environment state:
{env_desc}

Return ONLY the final answer, nothing else."""
    return system, user


def run_single(task: Task, model_name: str, budget_level: str, bap: bool = False) -> EvalResult:
    """Run a single task evaluation."""
    provider, model_id, is_base = MODEL_CONFIGS[model_name]
    client = get_client(provider)
    budget = BUDGET_LEVELS[budget_level]

    # Cap budget at model max (DashScope models have 8192 limit)
    max_tokens_limit = 8192 if provider == "dashscope" else budget
    effective_budget = min(budget, max_tokens_limit)

    system, user = build_prompt(task, budget, bap=bap)

    kwargs = {}
    if model_id.startswith("qwen3"):
        kwargs["extra_body"] = {"enable_thinking": False}

    try:
        resp = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=effective_budget,
            temperature=0.0,
            **kwargs,
        )
        response = resp.choices[0].message.content or ""
        tokens_used = resp.usage.completion_tokens if resp.usage else count_tokens(response)
    except Exception as e:
        response = f"ERROR: {e}"
        tokens_used = 0

    success = score_task(task, response)
    return EvalResult(
        task_id=task.id, model=model_name, budget_level=budget_level,
        budget_tokens=budget, success=success, tokens_used=tokens_used,
        response=response, timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
    )


def run_block1(models: list[str] = None, budget_levels: list[str] = None):
    """Run Block 1: Main evaluation."""
    tasks = load_tasks()
    models = models or list(MODEL_CONFIGS.keys())
    budget_levels = budget_levels or list(BUDGET_LEVELS.keys())
    results = []

    for model in models:
        print(f"\n{'='*60}\nModel: {model}\n{'='*60}")
        for bl in budget_levels:
            print(f"  Budget: {bl} ({BUDGET_LEVELS[bl]} tokens)")
            bl_results = []
            for task in tasks:
                r = run_single(task, model, bl)
                bl_results.append(r)
                results.append(r)
            success_rate = sum(r.success for r in bl_results) / len(bl_results)
            print(f"    success@{bl} = {success_rate:.3f}")
            time.sleep(0.5)  # rate limiting

        # Compute model-level metrics
        model_results = [r for r in results if r.model == model]
        eps = compute_elasticity(model_results)
        cliff, cliff_at = compute_cliff_index(model_results)
        print(f"  ε({model}) = {eps:.3f}, Cliff = {cliff:.3f} at {cliff_at}")

    save_results(results, "block1_results.json")
    print(f"\nBlock 1 complete. {len(results)} evaluations saved.")
    return results


def run_block2(models: list[str] = None):
    """Run Block 2: BAP intervention comparison."""
    tasks = load_tasks()
    models = models or ["gpt-4o-mini", "claude-3.5-haiku", "qwen2.5-7b", "deepseek-v3", "llama-3.1-8b-instruct", "gemini-2.0-flash"]
    results = []

    for model in models:
        print(f"\n{'='*60}\nModel: {model} (BAP)\n{'='*60}")
        for bl in BUDGET_LEVELS:
            standard_results, bap_results = [], []
            for task in tasks:
                r_std = run_single(task, model, bl, bap=False)
                r_bap = run_single(task, model, bl, bap=True)
                standard_results.append(r_std)
                bap_results.append(r_bap)
                results.extend([r_std, r_bap])
            std_rate = sum(r.success for r in standard_results) / len(standard_results)
            bap_rate = sum(r.success for r in bap_results) / len(bap_results)
            print(f"  {bl}: standard={std_rate:.3f}, BAP={bap_rate:.3f}, Δ={bap_rate-std_rate:+.3f}")
            time.sleep(0.5)

    save_results(results, "block2_results.json")
    print(f"\nBlock 2 complete. {len(results)} evaluations saved.")
    return results


def analyze_results(results_file: str = "block1_results.json"):
    """Analyze saved results and print summary."""
    results_path = Path(__file__).parent / "results" / results_file
    data = json.loads(results_path.read_text())
    results = [EvalResult(**d) for d in data]

    models = sorted(set(r.model for r in results))
    print(f"\n{'Model':<25} {'ε(m)':<8} {'Cliff':<8} {'B1':<6} {'B2':<6} {'B3':<6} {'B4':<6} {'B5':<6}")
    print("-" * 80)

    for model in models:
        mr = [r for r in results if r.model == model]
        eps = compute_elasticity(mr)
        cliff, _ = compute_cliff_index(mr)
        perfs = {}
        for bl in BUDGET_LEVELS:
            bl_r = [r for r in mr if r.budget_level == bl]
            perfs[bl] = sum(r.success for r in bl_r) / len(bl_r) if bl_r else 0
        print(f"{model:<25} {eps:<8.3f} {cliff:<8.3f} {perfs['B1']:<6.3f} {perfs['B2']:<6.3f} {perfs['B3']:<6.3f} {perfs['B4']:<6.3f} {perfs['B5']:<6.3f}")


if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")

    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "block1":
            models = sys.argv[2:] if len(sys.argv) > 2 else None
            run_block1(models=models)
        elif cmd == "block2":
            run_block2()
        elif cmd == "analyze":
            f = sys.argv[2] if len(sys.argv) > 2 else "block1_results.json"
            analyze_results(f)
    else:
        print("Usage: python -m budgetbench.runner [block1|block2|analyze] [models...]")
