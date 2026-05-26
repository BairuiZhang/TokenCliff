"""Generate multi-step tasks that are TRULY budget-sensitive.

Key insight: tasks must require VERBOSE intermediate reasoning that consumes tokens.
- At B1 (200 tokens): model must guess or skip steps → low accuracy
- At B3 (1500 tokens): model can do partial reasoning → medium accuracy  
- At B5 (8192 tokens): model can do full chain-of-thought → high accuracy

Strategy: tasks where the ENVIRONMENT DATA is large and requires careful extraction/computation.
The model needs to "show its work" to get the right answer.
"""
import json, random, string
from pathlib import Path

TASKS_DIR = Path(__file__).parent / "tasks"

def rand_name(n=5):
    return ''.join(random.choices(string.ascii_lowercase, k=n))


def gen_large_table_tasks():
    """Tasks with large tables where model must enumerate/filter many rows (50 tasks)."""
    tasks = []
    for i in range(50):
        random.seed(30000 + i)
        # Large table: 30-60 rows, multi-condition filter + aggregation
        n_rows = random.randint(30, 60)
        categories = ["alpha", "beta", "gamma", "delta", "epsilon"]
        statuses = ["active", "inactive", "pending"]
        regions = ["north", "south", "east", "west"]

        records = []
        for j in range(n_rows):
            records.append({
                "id": j + 1,
                "category": random.choice(categories),
                "status": random.choice(statuses),
                "region": random.choice(regions),
                "value": random.randint(10, 500),
                "priority": random.randint(1, 5)
            })

        # Complex multi-condition query
        qtype = i % 5
        if qtype == 0:
            target_cat = random.choice(categories)
            target_status = "active"
            filtered = [r for r in records if r["category"] == target_cat and r["status"] == target_status]
            answer = str(sum(r["value"] for r in filtered))
            question = f"Sum the 'value' of all records where category='{target_cat}' AND status='active'."
        elif qtype == 1:
            target_region = random.choice(regions)
            min_priority = random.randint(3, 4)
            filtered = [r for r in records if r["region"] == target_region and r["priority"] >= min_priority]
            answer = str(len(filtered))
            question = f"Count records where region='{target_region}' AND priority >= {min_priority}."
        elif qtype == 2:
            # Top-5 by value in a specific category, sum their priorities
            target_cat = random.choice(categories)
            cat_records = sorted([r for r in records if r["category"] == target_cat], key=lambda x: x["value"], reverse=True)
            top5 = cat_records[:5]
            answer = str(sum(r["priority"] for r in top5))
            question = f"Find the top 5 records by value in category '{target_cat}'. Return the sum of their priorities."
        elif qtype == 3:
            # Average value per region, find region with highest average
            region_vals = {}
            for r in records:
                region_vals.setdefault(r["region"], []).append(r["value"])
            region_avgs = {reg: sum(vals) / len(vals) for reg, vals in region_vals.items()}
            answer = max(region_avgs, key=region_avgs.get)
            question = "Calculate the average value for each region. Which region has the highest average? Return the region name."
        else:
            # Count unique categories that have at least N active records
            threshold = random.randint(3, 6)
            cat_active_counts = {}
            for r in records:
                if r["status"] == "active":
                    cat_active_counts[r["category"]] = cat_active_counts.get(r["category"], 0) + 1
            qualifying = [c for c, cnt in cat_active_counts.items() if cnt >= threshold]
            answer = str(len(qualifying))
            question = f"How many categories have at least {threshold} active records? Return the count."

        tasks.append({
            "id": f"multistep_{i+1:03d}",
            "domain": "multistep",
            "prompt": f"Given the following records table, answer: {question} Return ONLY the answer.",
            "tools": [{"name": "query", "desc": "Query records"}],
            "environment": {"tables": {"records": records}},
            "ground_truth": answer,
            "difficulty": 4
        })
    return tasks


def gen_multi_file_extraction():
    """Tasks requiring reading and combining info from multiple files (50 tasks)."""
    tasks = []
    for i in range(50):
        random.seed(31000 + i)
        n_files = random.randint(8, 15)
        files = {}

        # Each file contains a key-value pair buried in text
        keys_values = {}
        target_keys = []
        for j in range(n_files):
            name = f"/reports/{rand_name()}.txt"
            key = f"metric_{rand_name(3)}"
            value = random.randint(10, 999)
            keys_values[key] = value
            # Bury the value in verbose text
            filler = f"This quarterly report covers the period ending Q{random.randint(1,4)} 2026. " \
                     f"Performance indicators show mixed results across divisions. " \
                     f"The {key} measurement recorded a value of {value} this period. " \
                     f"Further analysis is needed to determine long-term trends."
            files[name] = filler
            if j < random.randint(3, 5):
                target_keys.append(key)

        # Question: sum/average specific metrics
        qtype = i % 3
        if qtype == 0:
            answer = str(sum(keys_values[k] for k in target_keys))
            keys_str = ", ".join(target_keys)
            question = f"Read all files in /reports. Find the values for these metrics: {keys_str}. Return their sum."
        elif qtype == 1:
            answer = str(max(keys_values[k] for k in target_keys))
            keys_str = ", ".join(target_keys)
            question = f"Read all files in /reports. Find the values for: {keys_str}. Return the maximum value."
        else:
            vals = [keys_values[k] for k in target_keys]
            answer = str(round(sum(vals) / len(vals)))
            keys_str = ", ".join(target_keys)
            question = f"Read all files in /reports. Find the values for: {keys_str}. Return their average (rounded to integer)."

        tasks.append({
            "id": f"multistep_{i+51:03d}",
            "domain": "multistep",
            "prompt": question,
            "tools": [{"name": "ls", "desc": "List directory"}, {"name": "read", "desc": "Read file"}],
            "environment": {"files": files},
            "ground_truth": answer,
            "difficulty": 4
        })
    return tasks


def gen_chain_computation():
    """Tasks requiring a chain of 4-6 arithmetic steps (60 tasks)."""
    tasks = []
    for i in range(60):
        random.seed(32000 + i)
        # Generate a computation chain that requires step-by-step work
        n_steps = random.randint(4, 6)
        values = [random.randint(10, 100) for _ in range(n_steps + 1)]
        ops = [random.choice(["+", "-", "*"]) for _ in range(n_steps)]

        # Compute result step by step
        result = values[0]
        steps_desc = [str(values[0])]
        for j in range(n_steps):
            if ops[j] == "+":
                result = result + values[j + 1]
            elif ops[j] == "-":
                result = result - values[j + 1]
            else:  # *
                result = result * values[j + 1]
            steps_desc.append(f"{ops[j]} {values[j + 1]}")

        chain = " ".join(steps_desc)
        # Add context that makes it harder to shortcut
        items = [{"name": rand_name(4), "amount": values[j]} for j in range(n_steps + 1)]
        operations = []
        for j in range(n_steps):
            if ops[j] == "+":
                operations.append(f"add item '{items[j+1]['name']}' ({items[j+1]['amount']})")
            elif ops[j] == "-":
                operations.append(f"subtract item '{items[j+1]['name']}' ({items[j+1]['amount']})")
            else:
                operations.append(f"multiply by item '{items[j+1]['name']}' ({items[j+1]['amount']})")

        question = f"Start with {items[0]['name']} = {items[0]['amount']}. Then: {'; '.join(operations)}. What is the final result?"

        tasks.append({
            "id": f"multistep_{i+101:03d}",
            "domain": "multistep",
            "prompt": question,
            "tools": [],
            "environment": {"events": [], "constraints": []},
            "ground_truth": str(result),
            "difficulty": 3 + (n_steps - 4)
        })
    return tasks


def generate_multistep():
    """Generate 160 multi-step tasks."""
    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    all_tasks = gen_large_table_tasks() + gen_multi_file_extraction() + gen_chain_computation()
    assert len(all_tasks) == 160
    for task in all_tasks:
        path = TASKS_DIR / f"{task['id']}.json"
        path.write_text(json.dumps(task, indent=2, ensure_ascii=False))
    print(f"Generated {len(all_tasks)} multi-step tasks in {TASKS_DIR}")
    return all_tasks


if __name__ == "__main__":
    generate_multistep()
