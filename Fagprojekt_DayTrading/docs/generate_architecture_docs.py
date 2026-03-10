from __future__ import annotations

import ast
import html
import json
import os
import re
import subprocess
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
PACKAGE_ROOT = SRC_ROOT / "kvant"
OUTPUT_DIR = PROJECT_ROOT / "docs" / "generated"


@dataclass
class FunctionInfo:
    name: str
    qualname: str
    signature: str
    lineno: int
    returns: str | None
    is_method: bool = False


@dataclass
class ClassInfo:
    name: str
    lineno: int
    methods: list[FunctionInfo] = field(default_factory=list)


@dataclass
class ModuleInfo:
    path: Path
    module_name: str
    package_group: str
    imports: set[str] = field(default_factory=set)
    internal_imports: set[str] = field(default_factory=set)
    external_imports: set[str] = field(default_factory=set)
    functions: list[FunctionInfo] = field(default_factory=list)
    classes: list[ClassInfo] = field(default_factory=list)
    docstring: str = ""
    has_main_guard: bool = False
    side_effects: set[str] = field(default_factory=set)
    summary: str = ""


def module_name_for(path: Path) -> str:
    rel = path.relative_to(SRC_ROOT).with_suffix("")
    return ".".join(rel.parts)


def package_group_for(module_name: str) -> str:
    parts = module_name.split(".")
    if module_name == "tasks":
        return "entry"
    if len(parts) < 2:
        return "core"
    if parts[1] == "kdata":
        return "data_ingestion"
    if parts[1] == "kmarket_info":
        return "market_info"
    if parts[1] == "ml_prepare_data":
        if len(parts) > 2 and parts[2] == "features":
            return "feature_engineering"
        if len(parts) > 2 and parts[2] == "plot_labelling":
            return "analysis_tools"
        return "data_preparation"
    if parts[1] == "ml_framework":
        if len(parts) > 2 and parts[2] == "logging":
            return "logging"
        if len(parts) > 2 and parts[2] == "models":
            return "ml_models"
        if len(parts) > 2 and parts[2] == "scripts":
            return "entry"
        return "training_evaluation"
    if parts[1] == "labelling":
        return "data_preparation"
    return "core"


GROUP_LABELS = {
    "entry": "Entry / Orchestration",
    "data_ingestion": "Data Ingestion",
    "market_info": "Market Calendar",
    "data_preparation": "Data Preparation",
    "feature_engineering": "Feature Engineering",
    "ml_models": "ML Models",
    "training_evaluation": "Training / Evaluation",
    "logging": "Logging / Monitoring",
    "analysis_tools": "Analysis Utilities",
    "core": "Core Package",
}


GROUP_COLORS = {
    "entry": "#1d4ed8",
    "data_ingestion": "#0f766e",
    "market_info": "#0f766e",
    "data_preparation": "#2563eb",
    "feature_engineering": "#7c3aed",
    "ml_models": "#b45309",
    "training_evaluation": "#dc2626",
    "logging": "#15803d",
    "analysis_tools": "#4338ca",
    "core": "#475569",
}


def short_annotation(node: ast.expr | None) -> str | None:
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except Exception:
        return None


def render_signature(node: ast.FunctionDef | ast.AsyncFunctionDef, class_name: str | None = None) -> str:
    pieces: list[str] = []
    posonly = list(node.args.posonlyargs)
    args = list(node.args.args)
    defaults = [None] * (len(args) - len(node.args.defaults)) + list(node.args.defaults)

    for idx, arg in enumerate(posonly):
        ann = short_annotation(arg.annotation)
        pieces.append(f"{arg.arg}: {ann}" if ann else arg.arg)
    if posonly:
        pieces.append("/")

    for arg, default in zip(args, defaults):
        if class_name and arg.arg == "self":
            pieces.append("self")
            continue
        ann = short_annotation(arg.annotation)
        chunk = f"{arg.arg}: {ann}" if ann else arg.arg
        if default is not None:
            try:
                chunk += f" = {ast.unparse(default)}"
            except Exception:
                chunk += " = ..."
        pieces.append(chunk)

    if node.args.vararg:
        ann = short_annotation(node.args.vararg.annotation)
        pieces.append(f"*{node.args.vararg.arg}: {ann}" if ann else f"*{node.args.vararg.arg}")
    elif node.args.kwonlyargs:
        pieces.append("*")

    for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
        ann = short_annotation(arg.annotation)
        chunk = f"{arg.arg}: {ann}" if ann else arg.arg
        if default is not None:
            try:
                chunk += f" = {ast.unparse(default)}"
            except Exception:
                chunk += " = ..."
        pieces.append(chunk)

    if node.args.kwarg:
        ann = short_annotation(node.args.kwarg.annotation)
        pieces.append(f"**{node.args.kwarg.arg}: {ann}" if ann else f"**{node.args.kwarg.arg}")

    ret = short_annotation(node.returns)
    return f"{node.name}({', '.join(pieces)})" + (f" -> {ret}" if ret else "")


def normalize_import(current_module: str, node: ast.ImportFrom) -> str | None:
    base_parts = current_module.split(".")[:-1]
    if node.level:
        keep = max(0, len(base_parts) - (node.level - 1))
        base_parts = base_parts[:keep]
    if node.module:
        return ".".join(base_parts + node.module.split("."))
    return ".".join(base_parts) if base_parts else None


def detect_side_effects(source: str) -> set[str]:
    effects: set[str] = set()
    patterns = {
        "filesystem reads": r"\b(read_text|open|np\.load|pq\.read_table|json\.loads)\b",
        "filesystem writes": r"\b(write_text|np\.save|mkdir|os\.makedirs|pq\.write_table)\b",
        "network / remote data": r"\b(hf_hub_download|wandb\.init|wandb\.log)\b",
        "model training": r"\b(loss\.backward|optimizer\.step|Trainer)\b",
        "visual reporting": r"\b(plt\.|wandb\.Image|report_sampling_density)\b",
        "stdout logging": r"\bprint\(",
    }
    for label, pattern in patterns.items():
        if re.search(pattern, source):
            effects.add(label)
    return effects


def summarize_module(module: ModuleInfo) -> str:
    name = module.path.stem
    if module.docstring:
        return module.docstring.strip().splitlines()[0].strip()
    heuristics = {
        "train_experiment": "CLI entrypoint for loading a prepared experiment, training the Conv1D classifier, and running final evaluation.",
        "prepare_experiment": "Builds prepared train/validation/test datasets from raw ticker data, sampling, feature engineering, and labeling.",
        "data_loading": "Loads a prepared experiment from disk and exposes PyTorch datasets and loaders for each split.",
        "trainer": "Runs epoch training loops, checkpoint selection, and periodic evaluation.",
        "evaluator": "Computes split-level metrics, per-ticker metrics, confusion matrices, and profit summaries.",
        "predict": "Runs batched inference and returns aligned predictions, labels, and sample identifiers.",
        "metrics": "Aggregates classification and profit-oriented evaluation metrics.",
        "wandb_logger": "Sends training metrics, confusion charts, and per-ticker summaries to Weights & Biases.",
        "conv1d": "Defines the Conv1D classification model used by the training pipeline.",
        "feature_engineering": "Creates OHLCV and technical indicator features, with optional standardization.",
        "tripple_bar": "Applies triple-barrier labeling and keeps per-sample metadata for later profit analysis.",
        "hf_minute_data": "Downloads, filters, and organizes minute-level OHLCV dataset splits from Hugging Face.",
        "hf_download_utils": "Loads monthly dataset shards from Hugging Face into pandas DataFrames.",
        "reporting": "Generates dataset sampling density diagnostics and reports.",
        "sampler_cumsum": "Implements the tuned CUSUM sampler used to reduce minute bars into event bars.",
        "sampling": "Defines the sampler protocol and simple baseline samplers.",
        "is_nyse_open": "Checks whether timestamps fall inside the NYSE trading window.",
        "labelling": "Provides the low-level triple-barrier labeling routine used by the labeler.",
    }
    if name in heuristics:
        return heuristics[name]
    if module.functions or module.classes:
        return f"Implements the {name.replace('_', ' ')} module in the {GROUP_LABELS.get(module.package_group, 'codebase')} layer."
    return f"Package marker for {module.module_name}."


def parse_module(path: Path) -> ModuleInfo:
    module_name = module_name_for(path)
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    info = ModuleInfo(
        path=path,
        module_name=module_name,
        package_group=package_group_for(module_name),
        docstring=ast.get_docstring(tree) or "",
    )

    for node in tree.body:
        if isinstance(node, ast.If):
            try:
                if ast.unparse(node.test) == "__name__ == '__main__'" or ast.unparse(node.test) == '__name__ == "__main__"':
                    info.has_main_guard = True
            except Exception:
                pass

        if isinstance(node, ast.Import):
            for alias in node.names:
                info.imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            normalized = normalize_import(module_name, node)
            if normalized:
                info.imports.add(normalized)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            info.functions.append(
                FunctionInfo(
                    name=node.name,
                    qualname=f"{module_name}.{node.name}",
                    signature=render_signature(node),
                    lineno=node.lineno,
                    returns=short_annotation(node.returns),
                )
            )
        elif isinstance(node, ast.ClassDef):
            class_info = ClassInfo(name=node.name, lineno=node.lineno)
            for body_node in node.body:
                if isinstance(body_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    class_info.methods.append(
                        FunctionInfo(
                            name=body_node.name,
                            qualname=f"{module_name}.{node.name}.{body_node.name}",
                            signature=render_signature(body_node, class_name=node.name),
                            lineno=body_node.lineno,
                            returns=short_annotation(body_node.returns),
                            is_method=True,
                        )
                    )
            info.classes.append(class_info)

    for imported in info.imports:
        if imported.startswith("kvant") or imported.startswith("src.kvant"):
            normalized = imported.replace("src.", "")
            info.internal_imports.add(normalized)
        else:
            info.external_imports.add(imported.split(".")[0])

    info.side_effects = detect_side_effects(source)
    info.summary = summarize_module(info)
    return info


def collect_modules() -> list[ModuleInfo]:
    modules = [parse_module(path) for path in sorted(PACKAGE_ROOT.rglob("*.py"))]
    tasks_path = PROJECT_ROOT / "tasks.py"
    if tasks_path.exists():
        source = tasks_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        info = ModuleInfo(
            path=tasks_path,
            module_name="tasks",
            package_group="entry",
            docstring=ast.get_docstring(tree) or "",
            side_effects=detect_side_effects(source),
        )
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                info.functions.append(
                    FunctionInfo(
                        name=node.name,
                        qualname=f"tasks.{node.name}",
                        signature=render_signature(node),
                        lineno=node.lineno,
                        returns=short_annotation(node.returns),
                    )
                )
        info.summary = "Invoke task entrypoints for testing, Docker builds, and docs commands."
        modules.append(info)
    return modules


def directory_tree(root: Path, max_depth: int = 4) -> str:
    lines = [root.name + "/"]

    def walk(path: Path, prefix: str = "", depth: int = 0) -> None:
        if depth >= max_depth:
            return
        children = [
            child
            for child in sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
            if child.name not in {".git", ".venv", "__pycache__", ".mypy_cache", ".pytest_cache"}
        ]
        for idx, child in enumerate(children):
            branch = "└── " if idx == len(children) - 1 else "├── "
            label = child.name + ("/" if child.is_dir() else "")
            lines.append(prefix + branch + label)
            if child.is_dir():
                extension = "    " if idx == len(children) - 1 else "│   "
                walk(child, prefix + extension, depth + 1)

    walk(root)
    return "\n".join(lines)


def find_entrypoints(modules: list[ModuleInfo]) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for module in modules:
        if module.has_main_guard:
            items.append((module.module_name, f"{module.path.relative_to(PROJECT_ROOT)} (__main__)"))
        if module.path.name == "train_experiment.py":
            items.append((module.module_name, "Training CLI"))
        if module.path.name == "prepare_experiment.py":
            items.append((module.module_name, "Data preparation CLI"))
        if module.path.name == "vary_labeller_runs.py":
            items.append((module.module_name, "Labelling sweep CLI"))
        if module.module_name == "tasks":
            items.append((module.module_name, "Invoke tasks"))
    deduped: list[tuple[str, str]] = []
    seen: set[str] = set()
    for name, desc in items:
        key = f"{name}:{desc}"
        if key not in seen:
            seen.add(key)
            deduped.append((name, desc))
    return deduped


def try_code2flow() -> dict[str, dict]:
    outputs: dict[str, dict] = {}
    jobs = {
        "training": "src/kvant/ml_framework/scripts/train_experiment.py",
        "data": "src/kvant/ml_prepare_data/prepare_experiment.py",
    }
    env = os.environ | {"UV_CACHE_DIR": "/tmp/uv-cache"}
    for key, source in jobs.items():
        out_path = Path("/tmp") / f"kvant_{key}_call_graph.json"
        cmd = [
            "uv",
            "run",
            "--with",
            "code2flow",
            "code2flow",
            source,
            "--output",
            str(out_path),
            "--language",
            "py",
            "--quiet",
        ]
        try:
            subprocess.run(
                cmd,
                cwd=PROJECT_ROOT,
                env=env,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            outputs[key] = json.loads(out_path.read_text(encoding="utf-8"))["graph"]
            out_path.unlink(missing_ok=True)
        except Exception:
            continue
    return outputs


def escape_xml(text: str) -> str:
    return html.escape(text, quote=True)


def estimate_box_height(lines: Iterable[str]) -> int:
    count = sum(1 for _ in lines)
    return 36 + (count * 16)


def wrap_lines(text: str, width: int) -> list[str]:
    return textwrap.wrap(text, width=width) or [text]


def box(node_id: str, x: int, y: int, w: int, h: int, title: str, subtitle: str, fill: str) -> str:
    parts = [
        f'<g id="{escape_xml(node_id)}">',
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" fill="{fill}" stroke="#0f172a" stroke-width="1.4"/>',
        f'<text x="{x + 18}" y="{y + 28}" font-family="Arial, Helvetica, sans-serif" font-size="16" font-weight="700" fill="#ffffff">{escape_xml(title)}</text>',
    ]
    sub_lines = wrap_lines(subtitle, 30)
    for idx, line in enumerate(sub_lines):
        parts.append(
            f'<text x="{x + 18}" y="{y + 52 + idx * 16}" font-family="Arial, Helvetica, sans-serif" font-size="12" fill="#e2e8f0">{escape_xml(line)}</text>'
        )
    parts.append("</g>")
    return "\n".join(parts)


def arrow(x1: int, y1: int, x2: int, y2: int, color: str = "#334155", label: str | None = None) -> str:
    mx = (x1 + x2) // 2
    path = f"M{x1},{y1} C{mx},{y1} {mx},{y2} {x2},{y2}"
    parts = [
        f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.2" marker-end="url(#arrowhead)"/>',
    ]
    if label:
        parts.append(
            f'<text x="{mx}" y="{min(y1, y2) - 8}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="11" fill="{color}">{escape_xml(label)}</text>'
        )
    return "\n".join(parts)


def svg_document(width: int, height: int, title: str, content: str) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{escape_xml(title)}">
<defs>
  <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
    <polygon points="0 0, 10 3.5, 0 7" fill="#334155"/>
  </marker>
</defs>
<rect width="100%" height="100%" fill="#f8fafc"/>
<text x="36" y="40" font-family="Arial, Helvetica, sans-serif" font-size="24" font-weight="700" fill="#0f172a">{escape_xml(title)}</text>
{content}
</svg>
"""


def write_svg(path: Path, title: str, nodes: list[dict], edges: list[tuple[str, str, str | None]], width: int = 1500) -> None:
    positions = {node["id"]: node for node in nodes}
    max_y = 100
    content: list[str] = []
    for src, dst, label in edges:
        a = positions[src]
        b = positions[dst]
        content.append(arrow(a["x"] + a["w"], a["y"] + a["h"] // 2, b["x"], b["y"] + b["h"] // 2, label=label))
    for node in nodes:
        content.append(box(node["id"], node["x"], node["y"], node["w"], node["h"], node["title"], node["subtitle"], node["fill"]))
        max_y = max(max_y, node["y"] + node["h"])
    path.write_text(svg_document(width, max_y + 60, title, "\n".join(content)), encoding="utf-8")


def build_architecture_svg(modules: list[ModuleInfo]) -> None:
    layers = [
        ("entry", ["tasks.py", "ml_framework/scripts/train_experiment.py", "ml_prepare_data/prepare_experiment.py"]),
        ("data_ingestion", ["kdata/hf_minute_data.py", "kdata/hf_download_utils.py", "kmarket_info/is_nyse_open.py"]),
        ("data_preparation", ["ml_prepare_data/samplers/*", "ml_prepare_data/labelling/tripple_bar.py", "ml_prepare_data/data_loading.py"]),
        ("feature_engineering", ["ml_prepare_data/features/feature_engineering.py"]),
        ("ml_models", ["ml_framework/models/conv1d.py"]),
        ("training_evaluation", ["ml_framework/train/*"]),
        ("logging", ["ml_framework/logging/wandb_logger.py"]),
    ]
    nodes = []
    x = 40
    for idx, (group, files) in enumerate(layers):
        nodes.append(
            {
                "id": group,
                "x": x + idx * 205,
                "y": 90 + (idx % 2) * 40,
                "w": 180,
                "h": 110,
                "title": GROUP_LABELS[group],
                "subtitle": ", ".join(files[:2]) + ("..." if len(files) > 2 else ""),
                "fill": GROUP_COLORS[group],
            }
        )
    edges = [
        ("entry", "data_ingestion", "raw data"),
        ("entry", "data_preparation", "prepare"),
        ("data_ingestion", "data_preparation", "OHLCV frames"),
        ("data_preparation", "feature_engineering", "sampled bars"),
        ("feature_engineering", "training_evaluation", "prepared tensors"),
        ("training_evaluation", "ml_models", "forward pass"),
        ("training_evaluation", "logging", "metrics"),
    ]
    write_svg(OUTPUT_DIR / "architecture_overview.svg", "Layered Architecture Overview", nodes, edges, width=1500)


def build_module_dependency_svg(modules: list[ModuleInfo]) -> None:
    selected = [
        ("kvant.kdata.hf_minute_data", "Dataset split builder"),
        ("kvant.ml_prepare_data.prepare_experiment", "Preparation orchestrator"),
        ("kvant.ml_prepare_data.features.feature_engineering", "Feature engineering"),
        ("kvant.ml_prepare_data.labelling.tripple_bar", "Triple-barrier labeler"),
        ("kvant.ml_prepare_data.data_loading", "Prepared dataset loader"),
        ("kvant.ml_framework.scripts.train_experiment", "Training CLI"),
        ("kvant.ml_framework.models.conv1d", "Conv1D model"),
        ("kvant.ml_framework.train.trainer", "Trainer"),
        ("kvant.ml_framework.train.evaluator", "Evaluator"),
        ("kvant.ml_framework.train.predict", "Prediction loop"),
        ("kvant.ml_framework.train.metrics", "Metric calculators"),
        ("kvant.ml_framework.logging.wandb_logger", "W&B logger"),
    ]
    nodes = []
    positions = [
        (40, 110), (280, 110), (520, 60), (520, 190),
        (760, 110), (1000, 60), (1240, 60), (1240, 190),
        (1480, 60), (1480, 190), (1720, 60), (1720, 190),
    ]
    for (module_name, subtitle), (x, y) in zip(selected, positions):
        group = next((m.package_group for m in modules if m.module_name == module_name), "core")
        nodes.append(
            {
                "id": module_name,
                "x": x,
                "y": y,
                "w": 210,
                "h": 96,
                "title": module_name.split(".")[-1],
                "subtitle": subtitle,
                "fill": GROUP_COLORS.get(group, "#475569"),
            }
        )
    edges = [
        ("kvant.kdata.hf_minute_data", "kvant.ml_prepare_data.prepare_experiment", "ticker splits"),
        ("kvant.ml_prepare_data.prepare_experiment", "kvant.ml_prepare_data.features.feature_engineering", "fit/transform"),
        ("kvant.ml_prepare_data.prepare_experiment", "kvant.ml_prepare_data.labelling.tripple_bar", "labels"),
        ("kvant.ml_prepare_data.prepare_experiment", "kvant.ml_prepare_data.data_loading", "artifacts"),
        ("kvant.ml_prepare_data.data_loading", "kvant.ml_framework.scripts.train_experiment", "PreparedExperiment"),
        ("kvant.ml_framework.scripts.train_experiment", "kvant.ml_framework.models.conv1d", "instantiate"),
        ("kvant.ml_framework.scripts.train_experiment", "kvant.ml_framework.train.trainer", "train loop"),
        ("kvant.ml_framework.scripts.train_experiment", "kvant.ml_framework.train.evaluator", "evaluation"),
        ("kvant.ml_framework.scripts.train_experiment", "kvant.ml_framework.logging.wandb_logger", "logging"),
        ("kvant.ml_framework.train.trainer", "kvant.ml_framework.train.evaluator", "periodic eval"),
        ("kvant.ml_framework.train.evaluator", "kvant.ml_framework.train.predict", "batched inference"),
        ("kvant.ml_framework.train.evaluator", "kvant.ml_framework.train.metrics", "aggregate metrics"),
        ("kvant.ml_framework.train.evaluator", "kvant.ml_framework.logging.wandb_logger", "emit charts"),
    ]
    write_svg(OUTPUT_DIR / "module_dependency_graph.svg", "Focused Module Dependency Graph", nodes, edges, width=1980)


def build_training_pipeline_svg() -> None:
    labels = [
        ("entry", "Entry point", "train_experiment.main"),
        ("load", "Load prepared data", "PreparedExperiment.get_loaders"),
        ("model", "Build model", "Conv1DClassifier + AdamW"),
        ("weights", "Compute class weights", "class_weights_from_dataset"),
        ("logger", "Setup logging", "WandbLogger.setup"),
        ("train", "Train epochs", "Trainer.fit / train_one_epoch"),
        ("eval", "Evaluate", "ExperimentEvaluator.evaluate_all"),
        ("done", "Persist results", "best checkpoint + wandb stop"),
    ]
    nodes = []
    for idx, (node_id, title, subtitle) in enumerate(labels):
        nodes.append(
            {
                "id": node_id,
                "x": 40 + idx * 190,
                "y": 105 + (idx % 2) * 35,
                "w": 170,
                "h": 88,
                "title": title,
                "subtitle": subtitle,
                "fill": ["#1d4ed8", "#2563eb", "#b45309", "#7c3aed", "#15803d", "#dc2626", "#0f766e", "#475569"][idx],
            }
        )
    edges = [(labels[idx][0], labels[idx + 1][0], None) for idx in range(len(labels) - 1)]
    write_svg(OUTPUT_DIR / "training_pipeline.svg", "Training Execution Pipeline", nodes, edges, width=1600)


def build_data_pipeline_svg() -> None:
    labels = [
        ("entry", "Entry point", "prepare_experiment.main"),
        ("download", "Resolve dataset split", "get_huggingface_top_20_normal_splits"),
        ("load", "Load ticker data", "get_ticker_data"),
        ("sample", "Fit and sample bars", "TunedCUSUMBarSampler.fit/transform"),
        ("features", "Engineer features", "IntradayTA10Features / StandardizedFeatures"),
        ("label", "Apply labels", "TripleBarrierLabeler.transform"),
        ("store", "Write prepared artifacts", "save_ticker_artifacts + indices"),
        ("report", "Report + bookmark", "report_sampling_density + last_experiment.txt"),
    ]
    nodes = []
    for idx, (node_id, title, subtitle) in enumerate(labels):
        nodes.append(
            {
                "id": node_id,
                "x": 40 + idx * 190,
                "y": 105 + ((idx + 1) % 2) * 35,
                "w": 170,
                "h": 88,
                "title": title,
                "subtitle": subtitle,
                "fill": ["#1d4ed8", "#0f766e", "#2563eb", "#7c3aed", "#4338ca", "#b45309", "#dc2626", "#475569"][idx],
            }
        )
    edges = [(labels[idx][0], labels[idx + 1][0], None) for idx in range(len(labels) - 1)]
    write_svg(OUTPUT_DIR / "data_pipeline.svg", "Data Preparation Pipeline", nodes, edges, width=1600)


def build_function_call_graph_svg() -> None:
    labels = [
        ("main", "train_experiment.main", "Orchestrates training"),
        ("args", "parse_args", "Resolve experiment directory"),
        ("loaders", "PreparedExperiment.get_loaders", "Build train/val/test DataLoaders"),
        ("setup", "WandbLogger.setup", "Register dataset metadata"),
        ("fit", "Trainer.fit", "Epoch loop and checkpointing"),
        ("epoch", "Trainer.train_one_epoch", "Forward/backward pass"),
        ("acc", "Trainer.accuracy_only", "Light validation check"),
        ("eval", "ExperimentEvaluator.evaluate_all", "Split-wide evaluation"),
        ("split", "ExperimentEvaluator.evaluate_split", "Per-split metrics"),
        ("pred", "predict", "Inference loop"),
        ("metrics", "classification_metrics", "Accuracy summary"),
        ("profit", "compute_action_profit_stats", "Per-action profit stats"),
    ]
    coords = [
        (40, 90), (260, 90), (480, 90), (700, 90),
        (920, 50), (1140, 10), (1140, 130), (1360, 50),
        (1580, 50), (1800, 10), (1800, 130), (2020, 70),
    ]
    nodes = []
    for (node_id, title, subtitle), (x, y) in zip(labels, coords):
        nodes.append(
            {
                "id": node_id,
                "x": x,
                "y": y,
                "w": 190,
                "h": 88,
                "title": title,
                "subtitle": subtitle,
                "fill": "#1e293b" if node_id in {"main", "fit", "eval"} else "#475569",
            }
        )
    edges = [
        ("main", "args", None),
        ("main", "loaders", None),
        ("main", "setup", None),
        ("main", "fit", None),
        ("fit", "epoch", "every epoch"),
        ("fit", "acc", "light eval"),
        ("fit", "eval", "full eval"),
        ("eval", "split", "for each split"),
        ("split", "pred", None),
        ("split", "metrics", None),
        ("split", "profit", None),
    ]
    write_svg(OUTPUT_DIR / "function_call_graph.svg", "Focused Training Function Call Graph", nodes, edges, width=2250)


def format_inputs(module: ModuleInfo) -> list[str]:
    items = []
    for fn in module.functions[:3]:
        items.append(fn.signature)
    for cls in module.classes[:2]:
        for method in cls.methods[:1]:
            items.append(f"{cls.name}.{method.signature}")
    if not items:
        items.append("Package import only")
    return items[:4]


def format_outputs(module: ModuleInfo) -> list[str]:
    items = []
    for fn in module.functions[:3]:
        items.append(f"{fn.name}: {fn.returns or 'untyped return'}")
    if module.side_effects:
        items.append("Side effects: " + ", ".join(sorted(module.side_effects)))
    if not items:
        items.append("Package marker with no direct runtime output.")
    return items[:4]


def key_functions(module: ModuleInfo) -> list[str]:
    items = [f"{fn.signature}" for fn in module.functions[:4]]
    for cls in module.classes[:2]:
        items.extend(f"{cls.name}.{method.signature}" for method in cls.methods[:2])
    return items[:6]


def markdown_for_modules(modules: list[ModuleInfo]) -> str:
    blocks = []
    for module in sorted(modules, key=lambda m: str(m.path.relative_to(PROJECT_ROOT))):
        rel = module.path.relative_to(PROJECT_ROOT)
        blocks.append(f"### File: `{rel.as_posix()}`")
        blocks.append("")
        blocks.append("**Purpose**")
        blocks.append(module.summary)
        blocks.append("")
        blocks.append("**Inputs**")
        for item in format_inputs(module):
            blocks.append(f"- `{item}`")
        ext = ", ".join(sorted(module.external_imports)[:8])
        blocks.append(f"- External dependencies: `{ext or 'none'}`")
        blocks.append("")
        blocks.append("**Outputs**")
        for item in format_outputs(module):
            blocks.append(f"- {item}")
        blocks.append("")
        blocks.append("**Key Functions**")
        for item in key_functions(module):
            blocks.append(f"- `{item}`")
        blocks.append("")
    return "\n".join(blocks)


def markdown_document(modules: list[ModuleInfo], entrypoints: list[tuple[str, str]], tree_text: str) -> str:
    grouped: dict[str, list[str]] = {}
    for module in modules:
        grouped.setdefault(module.package_group, []).append(module.module_name)

    overview = (
        "The repository is organized around a two-stage quant research workflow: "
        "a data preparation pipeline under `kvant.ml_prepare_data` and a model training/evaluation pipeline under "
        "`kvant.ml_framework`. Raw minute OHLCV data is sourced from Hugging Face dataset shards, transformed into "
        "prepared experiment artifacts on disk, then consumed by a PyTorch training loop with W&B-based reporting."
    )

    pipeline = [
        ("Data entry", "`src/kvant/ml_prepare_data/prepare_experiment.py:main()` discovers a dataset split and loads per-ticker OHLCV frames."),
        ("Sampling", "`TunedCUSUMBarSampler.fit/transform()` reduces dense minute data to event bars."),
        ("Features", "`IntradayTA10Features` and `StandardizedFeatures` convert bars into model-ready features."),
        ("Labels", "`TripleBarrierLabeler.transform()` assigns targets and metadata for downstream profit analysis."),
        ("Prepared artifacts", "`prepare_experiment()` writes ticker arrays plus split indices into `src/kvant/ml_framework/prepared/`."),
        ("Training entry", "`src/kvant/ml_framework/scripts/train_experiment.py:main()` reads the latest prepared experiment."),
        ("Training loop", "`Trainer.fit()` trains `Conv1DClassifier`, checkpoints by validation accuracy, and triggers periodic evaluation."),
        ("Evaluation and logging", "`ExperimentEvaluator` computes metrics and `WandbLogger` publishes split and ticker-level diagnostics."),
    ]

    lines = [
        "# Architecture Documentation",
        "",
        "## System Overview",
        "",
        overview,
        "",
        "## Repository Structure",
        "",
        "```text",
        tree_text,
        "```",
        "",
        "## Entry Points",
        "",
    ]
    for module_name, desc in entrypoints:
        lines.append(f"- `{module_name}`: {desc}")
    lines.extend(
        [
            "",
            "## Architecture Layers",
            "",
            "![Architecture Overview](architecture_overview.svg)",
            "",
        ]
    )
    for group in ["entry", "data_ingestion", "market_info", "data_preparation", "feature_engineering", "ml_models", "training_evaluation", "logging", "analysis_tools", "core"]:
        members = grouped.get(group)
        if not members:
            continue
        lines.append(f"### {GROUP_LABELS[group]}")
        lines.append("")
        lines.append(", ".join(f"`{name}`" for name in sorted(members)))
        lines.append("")
    lines.extend(
        [
            "## Dependency Graph",
            "",
            "![Module Dependency Graph](module_dependency_graph.svg)",
            "",
            "This graph is intentionally focused on the modules that shape the end-to-end training workflow. "
            "It omits package marker files and low-signal helper modules to keep the node count readable.",
            "",
            "## Execution Pipelines",
            "",
            "### Data Preparation Pipeline",
            "",
            "![Data Pipeline](data_pipeline.svg)",
            "",
            "### Training Pipeline",
            "",
            "![Training Pipeline](training_pipeline.svg)",
            "",
            "### Pipeline Trace",
            "",
        ]
    )
    for stage, desc in pipeline:
        lines.append(f"- **{stage}**: {desc}")
    lines.extend(
        [
            "",
            "## Function Call Graph",
            "",
            "![Function Call Graph](function_call_graph.svg)",
            "",
            "This is a focused call graph centered on the training entrypoint and the evaluation stack. "
            "Static tools were used to inspect the repository, but the final visualization is reduced to the cross-module calls that matter operationally.",
            "",
            "## File-by-File Documentation",
            "",
            markdown_for_modules(modules),
        ]
    )
    return "\n".join(lines)


def escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def markdown_to_plain_text(md: str) -> str:
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", md)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = text.replace("```text", "").replace("```", "")
    return text


def write_simple_pdf(path: Path, text: str) -> None:
    width = 595
    height = 842
    margin = 50
    font_size = 10
    line_height = 14
    max_chars = 92

    wrapped_lines: list[str] = []
    for raw_line in markdown_to_plain_text(text).splitlines():
        if not raw_line.strip():
            wrapped_lines.append("")
            continue
        for line in textwrap.wrap(raw_line, width=max_chars):
            wrapped_lines.append(line)

    pages: list[list[str]] = []
    lines_per_page = (height - 2 * margin) // line_height
    for idx in range(0, len(wrapped_lines), lines_per_page):
        pages.append(wrapped_lines[idx: idx + lines_per_page])

    objects: list[bytes] = []

    def add_object(content: bytes) -> int:
        objects.append(content)
        return len(objects)

    font_obj = add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    page_ids: list[int] = []
    content_ids: list[int] = []

    for page_lines in pages:
        y = height - margin
        stream_lines = [b"BT", f"/F1 {font_size} Tf".encode(), f"1 0 0 1 {margin} {y} Tm".encode()]
        first = True
        for line in page_lines:
            if not first:
                stream_lines.append(f"0 -{line_height} Td".encode())
            first = False
            stream_lines.append(f"({escape_pdf_text(line)}) Tj".encode())
        stream_lines.append(b"ET")
        stream = b"\n".join(stream_lines)
        content_id = add_object(f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream")
        content_ids.append(content_id)
        page_obj = add_object(
            f"<< /Type /Page /Parent 0 0 R /MediaBox [0 0 {width} {height}] /Resources << /Font << /F1 {font_obj} 0 R >> >> /Contents {content_id} 0 R >>".encode()
        )
        page_ids.append(page_obj)

    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    pages_obj = add_object(f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode())
    for page_id in page_ids:
        content = objects[page_id - 1].decode("latin1").replace("/Parent 0 0 R", f"/Parent {pages_obj} 0 R")
        objects[page_id - 1] = content.encode("latin1")

    catalog_obj = add_object(f"<< /Type /Catalog /Pages {pages_obj} 0 R >>".encode())

    result = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for idx, obj in enumerate(objects, start=1):
        offsets.append(len(result))
        result.extend(f"{idx} 0 obj\n".encode())
        result.extend(obj)
        result.extend(b"\nendobj\n")
    xref_offset = len(result)
    result.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    result.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        result.extend(f"{offset:010d} 00000 n \n".encode())
    result.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_obj} 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode()
    )
    path.write_bytes(result)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    modules = collect_modules()
    entrypoints = find_entrypoints(modules)
    tree_text = directory_tree(PROJECT_ROOT, max_depth=4)
    _ = try_code2flow()

    build_architecture_svg(modules)
    build_module_dependency_svg(modules)
    build_training_pipeline_svg()
    build_data_pipeline_svg()
    build_function_call_graph_svg()

    markdown = markdown_document(modules, entrypoints, tree_text)
    md_path = OUTPUT_DIR / "architecture_documentation.md"
    pdf_path = OUTPUT_DIR / "architecture_documentation.pdf"
    md_path.write_text(markdown, encoding="utf-8")
    write_simple_pdf(pdf_path, markdown)


if __name__ == "__main__":
    main()
