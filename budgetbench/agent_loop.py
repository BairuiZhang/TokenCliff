"""Agent loop: multi-turn interaction with budget enforcement."""
import json, re, time
from budgetbench import Task, EvalResult, BUDGET_LEVELS, count_tokens, score_task
from budgetbench.runner import get_client, MODEL_CONFIGS


MAX_TURNS = 20


def execute_action(action: str, task: Task) -> str:
    """Execute an agent action against the task's virtual environment. Returns observation."""
    action = action.strip()

    # Parse: USE: tool_name(args)
    m = re.match(r'USE:\s*(\w+)\((.*)\)', action, re.DOTALL)
    if not m:
        return "ERROR: Invalid action format. Use 'USE: tool(args)' or 'SUBMIT: answer'"

    tool_name = m.group(1)
    args_str = m.group(2).strip().strip('"\'')

    domain = task.domain
    env = task.environment

    # === File Operations ===
    if tool_name == "ls":
        files = env.get("files", {})
        path = args_str or "/"
        entries = [p for p in files.keys() if p.startswith(path)]
        return json.dumps(entries)

    elif tool_name == "read":
        files = env.get("files", {})
        content = files.get(args_str, files.get(f"/{args_str}", "FILE_NOT_FOUND"))
        return content[:2000]  # cap to avoid huge observations

    elif tool_name == "stat":
        files = env.get("files", {})
        content = files.get(args_str, files.get(f"/{args_str}", ""))
        if content:
            return json.dumps({"path": args_str, "size": len(content), "lines": content.count("\n") + 1})
        return "FILE_NOT_FOUND"

    elif tool_name == "find":
        files = env.get("files", {})
        return json.dumps([p for p in files if args_str in p])

    # === Data / Query ===
    elif tool_name == "query":
        tables = env.get("tables", {})
        try:
            args = json.loads(args_str) if args_str.startswith("{") else {"table": args_str}
        except:
            args = {"table": args_str}
        table_name = args.get("table", list(tables.keys())[0] if tables else "")
        rows = tables.get(table_name, [])
        # Apply where filter if provided
        where = args.get("where", {})
        if where and isinstance(where, dict):
            rows = [r for r in rows if all(r.get(k) == v for k, v in where.items())]
        # Apply select if provided
        select = args.get("select", None)
        if select == "count(*)" or select == "count":
            return json.dumps({"count": len(rows)})
        # Return rows (cap at 50 to avoid huge output)
        return json.dumps(rows[:50], ensure_ascii=False)

    elif tool_name == "list_tables":
        tables = env.get("tables", {})
        return json.dumps(list(tables.keys()))

    elif tool_name == "schema":
        tables = env.get("tables", {})
        table_name = args_str or (list(tables.keys())[0] if tables else "")
        rows = tables.get(table_name, [])
        if rows:
            return json.dumps({"columns": list(rows[0].keys()), "row_count": len(rows)})
        return "TABLE_NOT_FOUND"

    # === API / Tool Use ===
    elif tool_name == "call":
        endpoints = env.get("endpoints", {})
        try:
            args = json.loads(args_str) if args_str.startswith("{") else {"endpoint": args_str}
        except:
            args = {"endpoint": args_str}
        ep_name = args.get("endpoint", "")
        params = {k: v for k, v in args.items() if k != "endpoint"}
        ep = endpoints.get(ep_name, {})
        if not ep:
            return json.dumps({"error": f"Unknown endpoint: {ep_name}"})
        key = json.dumps(params, sort_keys=True) if params else ""
        responses = ep.get("responses", {})
        if key in responses:
            return json.dumps(responses[key])
        return json.dumps(ep.get("default_response", {"error": "no match"}))

    # === Fallback: try to match tool name to endpoint ===
    else:
        endpoints = env.get("endpoints", {})
        if tool_name in endpoints:
            ep = endpoints[tool_name]
            try:
                params = json.loads(args_str) if args_str.startswith("{") else {}
            except:
                params = {}
            key = json.dumps(params, sort_keys=True) if params else ""
            responses = ep.get("responses", {})
            if key in responses:
                return json.dumps(responses[key])
            return json.dumps(ep.get("default_response", {"error": "no match"}))
        return f"ERROR: Unknown tool '{tool_name}'"


def build_agent_prompt(task: Task, budget: int) -> str:
    """Build initial system prompt for agent loop."""
    tools_desc = "\n".join(f"  - {t['name']}({t.get('args', '...')}): {t['desc']}" for t in task.tools) if task.tools else "  (none — solve by reasoning)"

    # For data tasks, give schema hint but not full data
    env_hint = ""
    if "tables" in task.environment:
        tables = task.environment["tables"]
        for tname, rows in tables.items():
            if rows:
                env_hint += f"\n  Table '{tname}': {len(rows)} rows, columns: {list(rows[0].keys())}"
    elif "files" in task.environment:
        files = task.environment["files"]
        env_hint += f"\n  Filesystem: {len(files)} files"
    elif "endpoints" in task.environment:
        endpoints = task.environment["endpoints"]
        env_hint += f"\n  APIs available: {list(endpoints.keys())}"

    return f"""You are an agent solving a task. You have a total budget of {budget} output tokens across ALL turns.

Task: {task.prompt}

Available tools:
{tools_desc}

Environment:{env_hint}

Each turn, respond with EXACTLY ONE of:
  THINK: <your reasoning>
  USE: tool_name(arguments)
  SUBMIT: <your final answer>

Rules:
- USE calls a tool and returns its output. Arguments are strings or JSON.
- THINK lets you reason (costs tokens but no tool call).
- SUBMIT ends the task. Give ONLY the answer after SUBMIT:
- Be efficient — you have limited budget.
- For query tool: query(table_name) or query({{"table":"name"}})
- For API tools: call({{"endpoint":"name", "param":"value"}}) or tool_name({{"param":"value"}})"""


def run_agent_loop(task: Task, model_name: str, budget_level: str, bap: bool = False) -> EvalResult:
    """Run a multi-turn agent loop with budget enforcement."""
    provider, model_id, is_base = MODEL_CONFIGS[model_name]
    client = get_client(provider)
    budget = BUDGET_LEVELS[budget_level]

    # Cap at provider limits
    max_per_turn = min(budget, 8192 if provider == "dashscope" else budget)

    system_prompt = build_agent_prompt(task, budget)
    if bap:
        system_prompt += f"\n\nBUDGET WARNING: You only have {budget} tokens total. Plan carefully. Use minimal tokens per turn."

    messages = [{"role": "system", "content": system_prompt}]
    messages.append({"role": "user", "content": "Begin."})

    total_tokens_used = 0
    trajectory = []
    final_response = ""

    kwargs = {}
    if model_id.startswith("qwen3"):
        kwargs["extra_body"] = {"enable_thinking": False}

    for turn in range(MAX_TURNS):
        remaining = budget - total_tokens_used
        if remaining <= 0:
            break

        turn_max = min(remaining, max_per_turn, 500)  # cap per-turn to leave room for more turns

        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=turn_max,
                temperature=0.0,
                stop=["USE:", "THINK:", "SUBMIT:"] if False else None,  # let model output freely
                **kwargs,
            )
            output = resp.choices[0].message.content or ""
            turn_tokens = resp.usage.completion_tokens if resp.usage else count_tokens(output)
        except Exception as e:
            output = f"ERROR: {e}"
            turn_tokens = 0

        total_tokens_used += turn_tokens
        trajectory.append({"turn": turn + 1, "output": output, "tokens": turn_tokens})

        # Parse action — model may output THINK + USE or THINK + SUBMIT in one turn
        if "SUBMIT:" in output:
            final_response = output.split("SUBMIT:")[-1].strip().split("\n")[0].strip()
            break
        elif "USE:" in output:
            # Match USE: tool_name(...) with balanced parens
            use_start = output.index("USE:")
            paren_start = output.find("(", use_start)
            if paren_start == -1:
                messages.append({"role": "assistant", "content": output})
                messages.append({"role": "user", "content": "Invalid format. Use: USE: tool(args). Try again."})
            else:
                depth, idx = 1, paren_start + 1
                while idx < len(output) and depth > 0:
                    if output[idx] == "(": depth += 1
                    elif output[idx] == ")": depth -= 1
                    idx += 1
                if depth == 0:
                    tool_call = output[use_start + 4:idx].strip()  # "tool_name(...)"
                    action = "USE: " + tool_call
                    observation = execute_action(action, task)
                    messages.append({"role": "assistant", "content": output})
                    messages.append({"role": "user", "content": f"Observation: {observation}\n\nBudget remaining: ~{budget - total_tokens_used} tokens. Continue."})
                else:
                    messages.append({"role": "assistant", "content": output})
                    messages.append({"role": "user", "content": "Invalid format. Use: USE: tool(args). Try again."})
        elif "THINK:" in output:
            messages.append({"role": "assistant", "content": output})
            messages.append({"role": "user", "content": f"Budget remaining: ~{budget - total_tokens_used} tokens. Continue."})
        else:
            final_response = output.strip()
            break

    # If we ran out of budget without SUBMIT, use last output
    if not final_response and trajectory:
        final_response = trajectory[-1]["output"]

    success = score_task(task, final_response)
    return EvalResult(
        task_id=task.id, model=model_name, budget_level=budget_level,
        budget_tokens=budget, success=success, tokens_used=total_tokens_used,
        response=final_response, trajectory=trajectory,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
    )
