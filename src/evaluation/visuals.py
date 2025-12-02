"""
Visualization utilities for Student Success artifacts.

This module consumes the structured JSON artifacts emitted by the training
pipeline and produces presentation-ready plots for the executive deck/demo.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid")

ARTIFACT_DIR = Path("artifacts")
PLOTS_DIR = ARTIFACT_DIR / "plots"


def _load_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    with open(path, "r") as fh:
        return json.load(fh)


def _ensure_plot_dir() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _save_multi_format(fig: plt.Figure, stem: Path) -> None:
    png_path = stem.with_suffix(".png")
    svg_path = stem.with_suffix(".svg")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_f1(latest_run: Dict[str, Any]) -> Optional[Path]:
    test_results = latest_run.get("test_results")
    if not test_results:
        return None
    scores = test_results.get("test_per_class_f1")
    if not scores:
        return None
    class_names = latest_run.get(
        "class_names",
        [f"Class {idx}" for idx in range(len(scores))]
    )
    _ensure_plot_dir()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=class_names, y=scores, color="#4c72b0", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1 (Test Set)")
    for idx, val in enumerate(scores):
        ax.text(idx, val + 0.02, f"{val:.2f}", ha="center", va="bottom", fontsize=10)
    output_stem = PLOTS_DIR / "per_class_f1"
    _save_multi_format(fig, output_stem)
    return output_stem


def plot_confusion_matrix(confusion_payload: Dict[str, Any]) -> Optional[Path]:
    final_cm = confusion_payload.get("Final")
    if not final_cm:
        return None
    matrix = np.array(final_cm.get("matrix"))
    class_names = final_cm.get("class_names", [f"Class {i}" for i in range(matrix.shape[0])])
    _ensure_plot_dir()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (Test Set)")
    output_stem = PLOTS_DIR / "confusion_matrix"
    _save_multi_format(fig, output_stem)
    return output_stem


def plot_feature_importance(feature_payload: List[Dict[str, Any]], top_n: int = 12) -> Optional[Path]:
    if not feature_payload:
        return None
    top_features = feature_payload[:top_n]
    features = [row.get("Feature", "Unknown") for row in top_features][::-1]
    importances = [
        row.get("Importance_Pct", row.get("Importance", 0))
        for row in top_features
    ][::-1]
    _ensure_plot_dir()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(features, importances, color="#1f77b4")
    ax.set_xlabel("Relative Importance (%)")
    ax.set_title(f"Top {top_n} Features")
    for idx, val in enumerate(importances):
        ax.text(val + 0.5, idx, f"{val:.1f}%", va="center")
    output_stem = PLOTS_DIR / "feature_importance"
    _save_multi_format(fig, output_stem)
    return output_stem


def generate_all(artifact_dir: Path = ARTIFACT_DIR) -> Dict[str, Optional[Path]]:
    artifact_dir = Path(artifact_dir)
    latest_run = _load_json(artifact_dir / "latest_run.json")
    feature_importance = _load_json(artifact_dir / "feature_importance.json") or []
    confusion_matrices = _load_json(artifact_dir / "confusion_matrices.json") or {}
    
    if latest_run is None:
        raise FileNotFoundError(
            f"No latest_run.json found in {artifact_dir}. "
            "Run the training pipeline before generating visuals."
        )
    
    outputs: Dict[str, Optional[Path]] = {}
    outputs["per_class_f1"] = plot_per_class_f1(latest_run)
    outputs["confusion_matrix"] = plot_confusion_matrix(confusion_matrices)
    outputs["feature_importance"] = plot_feature_importance(feature_importance)
    return outputs


if __name__ == "__main__":
    paths = generate_all()
    for name, path in paths.items():
        if path:
            print(f"✅ Generated {name} plot at {path.with_suffix('.png')}")
        else:
            print(f"⚠️ Skipped {name} plot (missing data)")

