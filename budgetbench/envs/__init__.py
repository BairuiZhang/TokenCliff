"""Virtual environments for BudgetBench tasks. Pure Python, no external deps."""
import json, random
from typing import Any


class VirtualFS:
    """In-memory filesystem."""
    def __init__(self, state: dict):
        self.files = state.get("files", {})  # path -> content

    def ls(self, path: str = "/") -> list[str]:
        return [p for p in self.files if p.startswith(path) and p != path]

    def read(self, path: str) -> str:
        return self.files.get(path, "FILE_NOT_FOUND")

    def find(self, pattern: str) -> list[str]:
        return [p for p in self.files if pattern in p]

    def stat(self, path: str) -> dict:
        content = self.files.get(path, "")
        return {"path": path, "size": len(content), "lines": content.count("\n") + 1}


class VirtualDB:
    """Dict-based tables with simple query interface."""
    def __init__(self, state: dict):
        self.tables = state.get("tables", {})  # name -> list of dicts

    def query(self, table: str, where: dict = None, select: list = None, agg: str = None, group_by: str = None) -> Any:
        rows = self.tables.get(table, [])
        if where:
            rows = [r for r in rows if all(r.get(k) == v for k, v in where.items())]
        if group_by and agg:
            groups = {}
            for r in rows:
                key = r.get(group_by, "")
                groups.setdefault(key, []).append(r)
            if agg.startswith("avg:"):
                col = agg.split(":")[1]
                return {k: sum(r[col] for r in v) / len(v) for k, v in groups.items()}
            elif agg.startswith("sum:"):
                col = agg.split(":")[1]
                return {k: sum(r[col] for r in v) for k, v in groups.items()}
            elif agg == "count":
                return {k: len(v) for k, v in groups.items()}
        if select:
            rows = [{k: r[k] for k in select if k in r} for r in rows]
        return rows


class VirtualAPI:
    """Mock API endpoints returning deterministic responses."""
    def __init__(self, state: dict):
        self.endpoints = state.get("endpoints", {})  # name -> {params -> response}

    def call(self, endpoint: str, params: dict = None) -> dict:
        ep = self.endpoints.get(endpoint)
        if not ep:
            return {"error": f"Unknown endpoint: {endpoint}"}
        # Match params to stored responses
        key = json.dumps(params, sort_keys=True) if params else ""
        if key in ep.get("responses", {}):
            return ep["responses"][key]
        return ep.get("default_response", {"error": "No matching response"})


class VirtualCalendar:
    """Constraint-based scheduling environment."""
    def __init__(self, state: dict):
        self.events = state.get("events", [])  # list of {start, end, title}
        self.constraints = state.get("constraints", [])

    def get_events(self, date: str = None) -> list:
        if date:
            return [e for e in self.events if e.get("date") == date]
        return self.events

    def check_conflict(self, start: int, end: int, date: str) -> bool:
        for e in self.events:
            if e.get("date") == date:
                if not (end <= e["start"] or start >= e["end"]):
                    return True
        return False

    def find_slot(self, duration: int, date: str) -> list[tuple[int, int]]:
        slots = []
        for hour in range(9, 18):
            if hour + duration <= 18 and not self.check_conflict(hour, hour + duration, date):
                slots.append((hour, hour + duration))
        return slots


ENV_REGISTRY = {
    "file_ops": VirtualFS,
    "data_transform": VirtualDB,
    "tool_use": VirtualAPI,
    "planning": VirtualCalendar,
}

def create_env(domain: str, state: dict):
    """Create the appropriate virtual environment for a task domain."""
    cls = ENV_REGISTRY.get(domain)
    if cls:
        return cls(state)
    raise ValueError(f"Unknown domain: {domain}")
