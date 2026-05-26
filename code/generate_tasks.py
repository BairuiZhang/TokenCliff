"""Generate BudgetBench-500 task corpus. Run: python -m budgetbench.generate_tasks"""
import json, random, string, hashlib
from pathlib import Path

TASKS_DIR = Path(__file__).parent / "tasks"
TASKS_PER_DOMAIN = 125

def rand_name(n=6, seed=None):
    if seed is not None:
        random.seed(seed)
    return ''.join(random.choices(string.ascii_lowercase, k=n))

def rand_csv(rows=10, cols=3, headers=None):
    headers = headers or [f"col_{i}" for i in range(cols)]
    lines = [",".join(headers)]
    for _ in range(rows):
        lines.append(",".join(str(random.randint(1, 100)) for _ in range(cols)))
    return "\n".join(lines)

# ============ FILE OPS (125 tasks) ============
def gen_file_ops():
    tasks = []

    # Type 1: Find file with max/min property (30 tasks)
    for i in range(30):
        random.seed(1000 + i)
        n_files = random.randint(5, 20)
        files = {}
        props = []
        for j in range(n_files):
            name = f"/data/{rand_name()}.csv"
            rows = random.randint(3, 80)
            files[name] = rand_csv(rows=rows)
            props.append((name, rows))
        if i % 2 == 0:
            target = max(props, key=lambda x: x[1])
            q = "most"
        else:
            target = min(props, key=lambda x: x[1])
            q = "fewest"
        tasks.append({
            "id": f"file_ops_{i+1:03d}", "domain": "file_ops",
            "prompt": f"List all files in /data and find the one with the {q} lines. Return only the filename.",
            "tools": [{"name": "ls", "desc": "List directory"}, {"name": "read", "desc": "Read file"}, {"name": "stat", "desc": "Get file stats"}],
            "environment": {"files": files},
            "ground_truth": target[0].split("/")[-1], "difficulty": 2
        })

    # Type 2: Search for keyword in files (30 tasks)
    for i in range(30):
        random.seed(2000 + i)
        files = {}
        keyword = rand_name(random.randint(4, 7))
        n_files = random.randint(6, 20)
        target_idx = random.randint(0, n_files - 1)
        for j in range(n_files):
            name = f"/docs/{rand_name()}.txt"
            if j == target_idx:
                target_file = name
                files[name] = f"Report section {j}. Contains keyword {keyword} in paragraph."
            else:
                files[name] = f"Document {j} about {rand_name(8)}. No relevant matches here."
        tasks.append({
            "id": f"file_ops_{i+31:03d}", "domain": "file_ops",
            "prompt": f"Search all files in /docs for the keyword '{keyword}'. Return the filename containing it.",
            "tools": [{"name": "ls", "desc": "List directory"}, {"name": "read", "desc": "Read file"}, {"name": "find", "desc": "Find files"}],
            "environment": {"files": files},
            "ground_truth": target_file.split("/")[-1], "difficulty": 2
        })

    # Type 3: Count/aggregate across files (25 tasks)
    for i in range(25):
        random.seed(3000 + i)
        files = {}
        total = 0
        n_files = random.randint(3, 10)
        for j in range(n_files):
            rows = random.randint(5, 30)
            files[f"/data/part_{j}.csv"] = rand_csv(rows=rows)
            total += rows
        tasks.append({
            "id": f"file_ops_{i+61:03d}", "domain": "file_ops",
            "prompt": "Count the total number of data rows (excluding headers) across all CSV files in /data. Return just the number.",
            "tools": [{"name": "ls", "desc": "List directory"}, {"name": "read", "desc": "Read file"}],
            "environment": {"files": files},
            "ground_truth": str(total), "difficulty": 3
        })

    # Type 4: File size comparison (20 tasks)
    for i in range(20):
        random.seed(3500 + i)
        files = {}
        n_files = random.randint(4, 12)
        sizes = []
        for j in range(n_files):
            name = f"/logs/{rand_name()}.log"
            content = "x" * random.randint(50, 5000)
            files[name] = content
            sizes.append((name, len(content)))
        # Find top-k by size
        k = random.randint(2, 3)
        sorted_sizes = sorted(sizes, key=lambda x: x[1], reverse=True)
        answer = str(sum(s[1] for s in sorted_sizes[:k]))
        tasks.append({
            "id": f"file_ops_{i+86:03d}", "domain": "file_ops",
            "prompt": f"Find the {k} largest files in /logs by character count. Return the sum of their sizes.",
            "tools": [{"name": "ls", "desc": "List directory"}, {"name": "read", "desc": "Read file"}, {"name": "stat", "desc": "Get file stats"}],
            "environment": {"files": files},
            "ground_truth": answer, "difficulty": 3
        })

    # Type 5: Path navigation (20 tasks)
    for i in range(20):
        random.seed(4000 + i)
        files = {}
        depth = random.randint(2, 4)
        target_content = str(random.randint(1000, 9999))
        path_parts = [rand_name(4) for _ in range(depth)]
        target_path = "/" + "/".join(path_parts) + "/target.txt"
        files[target_path] = target_content
        # Add decoy files
        for j in range(random.randint(5, 15)):
            decoy_path = "/" + "/".join(random.choices(path_parts + [rand_name(4)], k=random.randint(1, 3))) + f"/{rand_name()}.txt"
            files[decoy_path] = str(random.randint(1, 999))
        tasks.append({
            "id": f"file_ops_{i+106:03d}", "domain": "file_ops",
            "prompt": f"Navigate the directory tree to find 'target.txt'. Return its content.",
            "tools": [{"name": "ls", "desc": "List directory"}, {"name": "read", "desc": "Read file"}],
            "environment": {"files": files},
            "ground_truth": target_content, "difficulty": 3
        })

    return tasks[:TASKS_PER_DOMAIN]

# ============ DATA TRANSFORM (125 tasks) ============
def gen_data_transform():
    tasks = []
    depts = ["engineering", "sales", "marketing", "hr", "finance", "legal", "ops"]
    cities = ["NYC", "London", "Tokyo", "Berlin", "Paris", "Sydney"]

    # Type 1: Filter and count (30 tasks)
    for i in range(30):
        random.seed(5000 + i)
        employees = [{"name": rand_name(8), "dept": random.choice(depts), "salary": random.randint(40000, 200000), "city": random.choice(cities)} for _ in range(random.randint(15, 50))]
        target_dept = random.choice(depts)
        count = sum(1 for e in employees if e["dept"] == target_dept)
        tasks.append({
            "id": f"data_transform_{i+1:03d}", "domain": "data_transform",
            "prompt": f"How many employees are in the '{target_dept}' department? Return just the number.",
            "tools": [{"name": "query", "desc": "Query table with where/select/agg/group_by"}],
            "environment": {"tables": {"employees": employees}},
            "ground_truth": str(count), "difficulty": 1
        })

    # Type 2: Aggregate by group (30 tasks)
    for i in range(30):
        random.seed(6000 + i)
        employees = [{"name": rand_name(8), "dept": random.choice(depts[:4]), "salary": random.randint(40000, 200000)} for _ in range(random.randint(20, 40))]
        dept_sums, dept_counts = {}, {}
        for e in employees:
            dept_sums[e["dept"]] = dept_sums.get(e["dept"], 0) + e["salary"]
            dept_counts[e["dept"]] = dept_counts.get(e["dept"], 0) + 1
        if i % 3 == 0:
            answer = max(dept_sums, key=lambda d: dept_sums[d] / dept_counts[d])
            q = "highest average salary"
        elif i % 3 == 1:
            answer = max(dept_counts, key=dept_counts.get)
            q = "most employees"
        else:
            answer = max(dept_sums, key=dept_sums.get)
            q = "highest total salary expenditure"
        tasks.append({
            "id": f"data_transform_{i+31:03d}", "domain": "data_transform",
            "prompt": f"Find the department with the {q}. Return only the department name.",
            "tools": [{"name": "query", "desc": "Query table with where/select/agg/group_by"}],
            "environment": {"tables": {"employees": employees}},
            "ground_truth": answer, "difficulty": 3
        })

    # Type 3: Multi-table (25 tasks)
    for i in range(25):
        random.seed(7000 + i)
        categories = ["A", "B", "C", "D"]
        products = [{"id": j, "name": f"product_{j}", "category": random.choice(categories), "price": random.randint(10, 300)} for j in range(15)]
        orders = [{"product_id": random.randint(0, 14), "quantity": random.randint(1, 10)} for _ in range(30)]
        rev_by_cat = {}
        for o in orders:
            p = products[o["product_id"]]
            rev_by_cat[p["category"]] = rev_by_cat.get(p["category"], 0) + p["price"] * o["quantity"]
        answer = max(rev_by_cat, key=rev_by_cat.get)
        tasks.append({
            "id": f"data_transform_{i+61:03d}", "domain": "data_transform",
            "prompt": "Using 'products' and 'orders' tables, find which category has the highest total revenue (price × quantity). Return the category letter.",
            "tools": [{"name": "query", "desc": "Query table"}],
            "environment": {"tables": {"products": products, "orders": orders}},
            "ground_truth": answer, "difficulty": 4
        })

    # Type 4: Filtering with multiple conditions (20 tasks)
    for i in range(20):
        random.seed(7500 + i)
        employees = [{"name": rand_name(8), "dept": random.choice(depts[:4]), "salary": random.randint(40000, 200000), "city": random.choice(cities[:4]), "years": random.randint(1, 20)} for _ in range(30)]
        threshold = random.choice([60000, 80000, 100000])
        target_city = random.choice(cities[:4])
        count = sum(1 for e in employees if e["salary"] > threshold and e["city"] == target_city)
        tasks.append({
            "id": f"data_transform_{i+86:03d}", "domain": "data_transform",
            "prompt": f"How many employees in '{target_city}' have salary above {threshold}? Return just the number.",
            "tools": [{"name": "query", "desc": "Query table"}],
            "environment": {"tables": {"employees": employees}},
            "ground_truth": str(count), "difficulty": 2
        })

    # Type 5: Sorting and ranking (20 tasks)
    for i in range(20):
        random.seed(8000 + i)
        items = [{"name": rand_name(6), "score": random.randint(1, 100), "category": random.choice(["X", "Y", "Z"])} for _ in range(20)]
        k = random.randint(3, 5)
        top_k = sorted(items, key=lambda x: x["score"], reverse=True)[:k]
        answer = str(sum(it["score"] for it in top_k))
        tasks.append({
            "id": f"data_transform_{i+106:03d}", "domain": "data_transform",
            "prompt": f"Find the top {k} items by score. Return the sum of their scores.",
            "tools": [{"name": "query", "desc": "Query table"}],
            "environment": {"tables": {"items": items}},
            "ground_truth": answer, "difficulty": 3
        })

    return tasks[:TASKS_PER_DOMAIN]

# ============ TOOL USE (125 tasks) ============
def gen_tool_use():
    tasks = []

    # Type 1: Sequential API calls (30 tasks)
    for i in range(30):
        random.seed(9000 + i)
        city = random.choice(["Tokyo", "London", "NYC", "Paris", "Berlin", "Sydney", "Mumbai", "Toronto", "Seoul", "Dubai"])
        temp = random.randint(-10, 45)
        pop = random.randint(500000, 20000000)
        result = round(temp / (pop / 1000000), 2)
        tasks.append({
            "id": f"tool_use_{i+1:03d}", "domain": "tool_use",
            "prompt": f"Get the temperature in {city} using weather API, then get its population using city_info API. Return temperature divided by population (in millions), rounded to 2 decimal places.",
            "tools": [{"name": "weather", "desc": "Get temperature"}, {"name": "city_info", "desc": "Get city info"}],
            "environment": {"endpoints": {
                "weather": {"responses": {json.dumps({"city": city}, sort_keys=True): {"temp": temp}}, "default_response": {"error": "not found"}},
                "city_info": {"responses": {json.dumps({"city": city}, sort_keys=True): {"population": pop}}, "default_response": {"error": "unknown"}}
            }},
            "ground_truth": str(result), "difficulty": 2
        })

    # Type 2: Conditional tool use (30 tasks)
    for i in range(30):
        random.seed(10000 + i)
        n_items = random.randint(5, 12)
        items = [{"name": rand_name(5), "price": random.randint(10, 500), "in_stock": random.choice([True, False])} for _ in range(n_items)]
        in_stock = [it for it in items if it["in_stock"]]
        if i % 3 == 0:
            target = min(in_stock, key=lambda x: x["price"]) if in_stock else None
            q = "cheapest item that is in stock"
        elif i % 3 == 1:
            target = max(in_stock, key=lambda x: x["price"]) if in_stock else None
            q = "most expensive item that is in stock"
        else:
            target = min(items, key=lambda x: x["price"])
            q = "cheapest item regardless of stock"
        tasks.append({
            "id": f"tool_use_{i+31:03d}", "domain": "tool_use",
            "prompt": f"Use the inventory API to list all items. Find the {q}. Return its name.",
            "tools": [{"name": "inventory", "desc": "List items with price and stock"}],
            "environment": {"endpoints": {"inventory": {"responses": {"": {"items": items}}, "default_response": {"items": items}}}},
            "ground_truth": target["name"] if target else "none", "difficulty": 2
        })

    # Type 3: Currency conversion (25 tasks)
    for i in range(25):
        random.seed(11000 + i)
        rates = {"USD_EUR": round(random.uniform(0.85, 0.95), 4), "USD_GBP": round(random.uniform(0.72, 0.82), 4),
                 "USD_JPY": round(random.uniform(110, 155), 2), "EUR_GBP": round(random.uniform(0.82, 0.90), 4)}
        pair = random.choice(list(rates.keys()))
        from_c, to_c = pair.split("_")
        amount = random.randint(100, 5000)
        result = round(amount * rates[pair], 2)
        tasks.append({
            "id": f"tool_use_{i+61:03d}", "domain": "tool_use",
            "prompt": f"Use exchange_rate API to get {from_c} to {to_c} rate. Convert {amount} {from_c}. Return result rounded to 2 decimals.",
            "tools": [{"name": "exchange_rate", "desc": "Get exchange rate"}],
            "environment": {"endpoints": {"exchange_rate": {"responses": {json.dumps({"from": from_c, "to": to_c}, sort_keys=True): {"rate": rates[pair]}}, "default_response": {"error": "pair not found"}}}},
            "ground_truth": str(result), "difficulty": 2
        })

    # Type 4: Multi-step with branching (20 tasks)
    for i in range(20):
        random.seed(12000 + i)
        user_id = random.randint(1000, 9999)
        balance = random.randint(100, 10000)
        purchase = random.randint(50, 5000)
        can_afford = balance >= purchase
        tasks.append({
            "id": f"tool_use_{i+86:03d}", "domain": "tool_use",
            "prompt": f"Check user {user_id}'s balance using account API. Can they afford a purchase of ${purchase}? Return 'yes' or 'no'.",
            "tools": [{"name": "account", "desc": "Get user account info"}],
            "environment": {"endpoints": {"account": {"responses": {json.dumps({"user_id": user_id}, sort_keys=True): {"balance": balance}}, "default_response": {"error": "user not found"}}}},
            "ground_truth": "yes" if can_afford else "no", "difficulty": 2
        })

    # Type 5: Chained computation (20 tasks)
    for i in range(20):
        random.seed(13000 + i)
        a = random.randint(10, 100)
        b = random.randint(2, 20)
        op = random.choice(["multiply", "add", "subtract"])
        if op == "multiply": result = a * b
        elif op == "add": result = a + b
        else: result = a - b
        discount = random.randint(5, 30)
        final = round(result * (1 - discount / 100), 2)
        tasks.append({
            "id": f"tool_use_{i+106:03d}", "domain": "tool_use",
            "prompt": f"Use calculator API to {op} {a} and {b}, then apply a {discount}% discount. Return the final value rounded to 2 decimals.",
            "tools": [{"name": "calculator", "desc": "Perform arithmetic"}, {"name": "discount", "desc": "Apply discount"}],
            "environment": {"endpoints": {
                "calculator": {"responses": {json.dumps({"a": a, "b": b, "op": op}, sort_keys=True): {"result": result}}, "default_response": {"error": "invalid"}},
                "discount": {"responses": {json.dumps({"amount": result, "percent": discount}, sort_keys=True): {"final": final}}, "default_response": {"error": "invalid"}}
            }},
            "ground_truth": str(final), "difficulty": 3
        })

    return tasks[:TASKS_PER_DOMAIN]

# ============ PLANNING (125 tasks) ============
def gen_planning():
    tasks = []

    # Type 1: Scheduling (30 tasks)
    for i in range(30):
        random.seed(14000 + i)
        existing = []
        for _ in range(random.randint(2, 7)):
            start = random.randint(9, 16)
            existing.append({"date": "2026-01-15", "start": start, "end": start + 1, "title": f"mtg_{rand_name(3)}"})
        duration = random.choice([1, 2])
        slots = []
        for hour in range(9, 18 - duration + 1):
            conflict = any(not (hour + duration <= e["start"] or hour >= e["end"]) for e in existing)
            if not conflict:
                slots.append(hour)
                break
        answer = str(slots[0]) if slots else "none"
        tasks.append({
            "id": f"planning_{i+1:03d}", "domain": "planning",
            "prompt": f"Find the earliest {duration}-hour slot available between 9:00-18:00 on 2026-01-15. Return the start hour.",
            "tools": [{"name": "get_events", "desc": "Get events"}, {"name": "check_conflict", "desc": "Check conflicts"}, {"name": "find_slot", "desc": "Find slots"}],
            "environment": {"events": existing, "constraints": []},
            "ground_truth": answer, "difficulty": 2
        })

    # Type 2: Knapsack/optimization (30 tasks)
    for i in range(30):
        random.seed(15000 + i)
        n_items = random.randint(5, 10)
        items = [{"name": f"item_{j}", "weight": random.randint(1, 15), "value": random.randint(5, 80)} for j in range(n_items)]
        capacity = random.randint(15, 40)
        # Greedy by value/weight
        sorted_items = sorted(items, key=lambda x: x["value"] / x["weight"], reverse=True)
        selected, remaining = [], capacity
        for it in sorted_items:
            if it["weight"] <= remaining:
                selected.append(it["name"])
                remaining -= it["weight"]
        total_value = sum(it["value"] for it in items if it["name"] in selected)
        tasks.append({
            "id": f"planning_{i+31:03d}", "domain": "planning",
            "prompt": f"Knapsack capacity={capacity}. Items: {json.dumps(items)}. Maximize value using greedy (best value/weight ratio first). Return total value.",
            "tools": [],
            "environment": {"events": [], "constraints": []},
            "ground_truth": str(total_value), "difficulty": 4
        })

    # Type 3: Dependency/topological (25 tasks)
    for i in range(25):
        random.seed(16000 + i)
        n = random.randint(4, 8)
        task_names = [f"T{j}" for j in range(n)]
        deps = {}
        for j in range(1, n):
            nd = random.randint(0, min(2, j))
            deps[task_names[j]] = random.sample(task_names[:j], nd)
        def critical_path(deps, tasks):
            memo = {}
            def depth(t):
                if t in memo: return memo[t]
                memo[t] = 1 + max((depth(d) for d in deps.get(t, [])), default=0)
                return memo[t]
            return max(depth(t) for t in tasks)
        cp = critical_path(deps, task_names)
        tasks.append({
            "id": f"planning_{i+61:03d}", "domain": "planning",
            "prompt": f"Tasks with dependencies: {json.dumps(deps)}. Minimum sequential steps to complete all (parallel allowed)? Return the number.",
            "tools": [],
            "environment": {"events": [], "constraints": []},
            "ground_truth": str(cp), "difficulty": 4
        })

    # Type 4: Shortest path (20 tasks)
    for i in range(20):
        random.seed(17000 + i)
        nodes = list("ABCDEFGH"[:random.randint(4, 7)])
        edges = {}
        for _ in range(len(nodes) * 2):
            a, b = random.sample(nodes, 2)
            w = random.randint(1, 10)
            edges[f"{a}-{b}"] = w
            edges[f"{b}-{a}"] = w
        src, dst = nodes[0], nodes[-1]
        # BFS/Dijkstra for shortest
        import heapq
        dist = {n: float('inf') for n in nodes}
        dist[src] = 0
        pq = [(0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]: continue
            for edge, w in edges.items():
                a, b = edge.split("-")
                if a == u and dist[u] + w < dist[b]:
                    dist[b] = dist[u] + w
                    heapq.heappush(pq, (dist[b], b))
        answer = str(dist[dst]) if dist[dst] != float('inf') else "unreachable"
        tasks.append({
            "id": f"planning_{i+86:03d}", "domain": "planning",
            "prompt": f"Graph edges (node-node: weight): {json.dumps(edges)}. Shortest path from {src} to {dst}? Return the total weight.",
            "tools": [],
            "environment": {"events": [], "constraints": []},
            "ground_truth": answer, "difficulty": 5
        })

    # Type 5: Resource allocation (20 tasks)
    for i in range(20):
        random.seed(18000 + i)
        n_tasks = random.randint(3, 6)
        task_list = [{"name": f"job_{j}", "duration": random.randint(1, 5), "priority": random.randint(1, 10)} for j in range(n_tasks)]
        deadline = random.randint(5, 12)
        # Greedy by priority: pick highest priority tasks that fit
        sorted_tasks = sorted(task_list, key=lambda x: x["priority"], reverse=True)
        selected, time_used = [], 0
        for t in sorted_tasks:
            if time_used + t["duration"] <= deadline:
                selected.append(t["name"])
                time_used += t["duration"]
        answer = str(sum(t["priority"] for t in task_list if t["name"] in selected))
        tasks.append({
            "id": f"planning_{i+106:03d}", "domain": "planning",
            "prompt": f"Deadline={deadline} hours. Jobs: {json.dumps(task_list)}. Schedule by highest priority first. Return total priority of completed jobs.",
            "tools": [],
            "environment": {"events": [], "constraints": []},
            "ground_truth": answer, "difficulty": 3
        })

    return tasks[:TASKS_PER_DOMAIN]


def generate_all():
    """Generate all 500 tasks and save to JSON files."""
    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    # Clear old tasks
    for f in TASKS_DIR.glob("*.json"):
        f.unlink()
    all_tasks = gen_file_ops() + gen_data_transform() + gen_tool_use() + gen_planning()
    assert len(all_tasks) == 500, f"Expected 500 tasks, got {len(all_tasks)}"
    for task in all_tasks:
        path = TASKS_DIR / f"{task['id']}.json"
        path.write_text(json.dumps(task, indent=2, ensure_ascii=False))
    print(f"Generated {len(all_tasks)} tasks in {TASKS_DIR}")
    # Stats
    domains = {}
    for t in all_tasks:
        domains[t["domain"]] = domains.get(t["domain"], 0) + 1
    for d, c in sorted(domains.items()):
        print(f"  {d}: {c}")
    return all_tasks


if __name__ == "__main__":
    generate_all()
