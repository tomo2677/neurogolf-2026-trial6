from __future__ import annotations

import importlib.util
import json
import math
import sys
import types
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import onnx


ROOT = Path(__file__).resolve().parent
GRID_SHAPE = [1, 10, 30, 30]
DATA_TYPE = onnx.TensorProto.FLOAT
OPSET_IMPORTS = [onnx.helper.make_opsetid("", 10)]
IR_VERSION = 10
LEDGER_COLUMNS = [
    "task",
    "status",
    "local_points",
    "memory_bytes_approx",
    "params",
    "updated_at",
]
STATUS_VALUES = {
    "no_solution",
    "generated",
    "build_failed",
    "score_failed",
    "rule_invalid",
    "passes_local",
    "fails_local",
    "needs_human_review",
}


@dataclass
class OfficialUtils:
    module: Any
    source: str


def normalize_task_id(raw: str | int) -> str:
    text = str(raw).strip().lower().replace(".py", "").replace(".onnx", "")
    if text.startswith("task_"):
        text = text.removeprefix("task_")
    elif text.startswith("task"):
        text = text.removeprefix("task")
    if not text.isdigit():
        raise ValueError(f"Invalid task id: {raw!r}")
    return f"task{int(text):03d}"


def task_number(task: str | int) -> int:
    return int(normalize_task_id(task).removeprefix("task"))


def task_json_path(task: str | int) -> Path:
    return ROOT / "data" / f"{normalize_task_id(task)}.json"


def task_spec_path(task: str | int) -> Path:
    task_id = normalize_task_id(task)
    direct = ROOT / "task_specs" / f"{task_id}.md"
    if direct.exists():
        return direct
    number = task_number(task)
    underscored = ROOT / "task_specs" / f"task_{number}.md"
    if underscored.exists():
        return underscored
    return direct


def solution_path(task: str | int) -> Path:
    return ROOT / "solutions" / f"{normalize_task_id(task)}.py"


def onnx_path(task: str | int) -> Path:
    return ROOT / "outputs" / "onnx" / f"{normalize_task_id(task)}.onnx"


def report_path(task: str | int) -> Path:
    return ROOT / "outputs" / "reports" / f"{normalize_task_id(task)}_score.json"


def load_task(task: str | int) -> dict[str, Any]:
    path = task_json_path(task)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_examples(task_data: dict[str, Any]):
    for split in ("train", "test", "arc-gen"):
        for index, example in enumerate(task_data.get(split, [])):
            yield split, index, example


def grid_over_30(grid: list[list[int]]) -> bool:
    if not grid:
        return False
    return len(grid) > 30 or max((len(row) for row in grid), default=0) > 30


def example_over_30(example: dict[str, Any]) -> bool:
    return grid_over_30(example["input"]) or grid_over_30(example["output"])


def grid_to_one_hot(grid: list[list[int]]) -> np.ndarray:
    if grid_over_30(grid):
        raise ValueError("Grid exceeds 30x30")
    tensor = np.zeros(GRID_SHAPE, dtype=np.float32)
    for row_index, row in enumerate(grid):
        for col_index, color in enumerate(row):
            if not 0 <= int(color) < GRID_SHAPE[1]:
                raise ValueError(f"Color out of range 0..9: {color}")
            tensor[0, int(color), row_index, col_index] = 1.0
    return tensor


def binary_tensor_to_grid(tensor: np.ndarray, height: int, width: int) -> list[list[int]]:
    binary = (tensor > 0.0).astype(np.float32)
    grid: list[list[int]] = []
    for row in range(height):
        cells: list[int] = []
        for col in range(width):
            active = np.flatnonzero(binary[0, :, row, col] > 0.0)
            if len(active) == 1:
                cells.append(int(active[0]))
            elif len(active) == 0:
                cells.append(-1)
            else:
                cells.append(10)
        grid.append(cells)
    return grid


def make_io_value_infos():
    input_info = onnx.helper.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    output_info = onnx.helper.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    return input_info, output_info


def check_and_save_model(model: onnx.ModelProto, path: Path) -> int:
    onnx.checker.check_model(model, full_check=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, path)
    return path.stat().st_size


def local_points(memory_bytes_approx: int | None, params: int | None) -> float | None:
    if memory_bytes_approx is None or params is None:
        return None
    return max(1.0, 25.0 - math.log(max(1.0, memory_bytes_approx + params)))


def count_params_approx(model: onnx.ModelProto) -> int | None:
    params = 0
    for init in model.graph.initializer:
        if any(dim <= 0 for dim in init.dims):
            return None
        params += math.prod(init.dims)
    for sparse_init in model.graph.sparse_initializer:
        if any(dim <= 0 for dim in sparse_init.values.dims):
            return None
        params += math.prod(sparse_init.values.dims)
    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.name == "value":
                if any(dim <= 0 for dim in attr.t.dims):
                    return None
                params += math.prod(attr.t.dims)
            elif attr.name == "value_floats":
                params += len(attr.floats)
            elif attr.name == "value_ints":
                params += len(attr.ints)
            elif attr.name == "value_strings":
                params += len(attr.strings)
    return params


def _install_display_stubs() -> None:
    if "IPython" not in sys.modules:
        ipython = types.ModuleType("IPython")
        sys.modules["IPython"] = ipython
    else:
        ipython = sys.modules["IPython"]
    if "IPython.display" not in sys.modules:
        display_module = types.ModuleType("IPython.display")
        display_module.display = lambda *args, **kwargs: None
        display_module.FileLink = lambda filename: filename
        sys.modules["IPython.display"] = display_module
    ipython.display = sys.modules["IPython.display"]

    if "matplotlib" not in sys.modules:
        matplotlib = types.ModuleType("matplotlib")
        sys.modules["matplotlib"] = matplotlib
    else:
        matplotlib = sys.modules["matplotlib"]
    if "matplotlib.pyplot" not in sys.modules:
        pyplot = types.ModuleType("matplotlib.pyplot")
        pyplot.figure = lambda *args, **kwargs: None
        sys.modules["matplotlib.pyplot"] = pyplot
    matplotlib.pyplot = sys.modules["matplotlib.pyplot"]


def load_official_utils() -> OfficialUtils | None:
    path = ROOT / "data" / "neurogolf_utils" / "neurogolf_utils.py"
    if not path.exists():
        return None
    _install_display_stubs()
    spec = importlib.util.spec_from_file_location("local_neurogolf_utils", path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:
        return None
    required = ("sanitize_model", "run_network", "score_network")
    if not all(hasattr(module, name) for name in required):
        return None
    return OfficialUtils(module=module, source=str(path.relative_to(ROOT)))


def load_ledger() -> dict[str, dict[str, Any]]:
    path = ROOT / "task_ledger.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("task_ledger.json must be a task-keyed object")
    return data


def ledger_entry(task: str | int) -> dict[str, Any]:
    return {
        "task": normalize_task_id(task),
        "status": "no_solution",
        "local_points": None,
        "memory_bytes_approx": None,
        "params": None,
        "updated_at": None,
    }


def initialize_ledger(existing: dict[str, dict[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
    ledger = dict(existing or {})
    for spec_path in sorted((ROOT / "task_specs").glob("task*.md")):
        try:
            task_id = normalize_task_id(spec_path.stem)
        except ValueError:
            continue
        ledger.setdefault(task_id, ledger_entry(task_id))
    return ledger


def update_ledger(task: str | int, **updates: Any) -> dict[str, dict[str, Any]]:
    task_id = normalize_task_id(task)
    ledger = initialize_ledger(load_ledger())
    entry = ledger.setdefault(task_id, ledger_entry(task_id))
    for key, value in updates.items():
        if key not in LEDGER_COLUMNS:
            raise KeyError(f"Unexpected ledger column: {key}")
        if key == "status" and value not in STATUS_VALUES:
            raise ValueError(f"Invalid status: {value}")
        entry[key] = value
    entry["task"] = task_id
    entry.setdefault("updated_at", None)
    write_ledger(ledger)
    return ledger


def write_ledger(ledger: dict[str, dict[str, Any]]) -> None:
    json_path = ROOT / "task_ledger.json"
    normalized = {task: {col: entry.get(col) for col in LEDGER_COLUMNS} for task, entry in sorted(ledger.items())}
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(normalized, f, indent=2, ensure_ascii=False)
        f.write("\n")
    write_ledger_markdown(normalized)


def write_ledger_markdown(ledger: dict[str, dict[str, Any]]) -> None:
    md_path = ROOT / "task_ledger.md"
    header = "| " + " | ".join(LEDGER_COLUMNS) + " |"
    sep = "| " + " | ".join(["---"] * len(LEDGER_COLUMNS)) + " |"
    lines = [header, sep]
    for task, entry in sorted(ledger.items()):
        row = []
        for col in LEDGER_COLUMNS:
            value = entry.get(col)
            row.append("" if value is None else str(value))
        lines.append("| " + " | ".join(row) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def utc_timestamp() -> str:
    return datetime.now().astimezone().replace(microsecond=0).isoformat()
