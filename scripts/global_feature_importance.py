"""Permutation importance of the 16 production features on validation PR-AUC.

Permutes one original raw column at a time (before preprocessing), runs the
full frozen pipeline (preprocess → MLP → calibrate), and measures the mean
decrease in validation PR-AUC over 20 deterministic repeats.

Does not retrain, modify, or promote the model. Does not access the test set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from conversion_router.data.load import load_raw_dataset
from conversion_router.data.preprocess import (
    BOOLEAN_COLUMNS,
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
)
from conversion_router.data.split import load_split_indices
from conversion_router.modeling.calibrate import apply_calibration
from conversion_router.modeling.inference import load_artifacts
from conversion_router.modeling.network import predict_proba

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = PROJECT_ROOT / "docs" / "global_feature_importance.json"
FIGURE_PATH = PROJECT_ROOT / "docs" / "figures" / "global_feature_importance.png"

SEED = 42
N_REPEATS = 20

ALL_FEATURES = list(NUMERIC_COLUMNS) + list(BOOLEAN_COLUMNS) + list(CATEGORICAL_COLUMNS)

TURKISH_LABELS: dict[str, str] = {
    "Administrative": "Yönetim sayfası sayısı (Administrative)",
    "Administrative_Duration": "Yönetim sayfası süresi (Administrative_Duration)",
    "Informational": "Bilgi sayfası sayısı (Informational)",
    "Informational_Duration": "Bilgi sayfası süresi (Informational_Duration)",
    "ProductRelated": "Ürün sayfası sayısı (ProductRelated)",
    "ProductRelated_Duration": "Ürün sayfası süresi (ProductRelated_Duration)",
    "BounceRates": "Hemen çıkma oranı (BounceRates)",
    "ExitRates": "Çıkış oranı (ExitRates)",
    "SpecialDay": "Özel gün yakınlığı (SpecialDay)",
    "Month": "Ay (Month)",
    "OperatingSystems": "İşletim sistemi (OperatingSystems)",
    "Browser": "Tarayıcı (Browser)",
    "Region": "Bölge (Region)",
    "TrafficType": "Trafik türü (TrafficType)",
    "VisitorType": "Ziyaretçi türü (VisitorType)",
    "Weekend": "Hafta sonu (Weekend)",
}


def _score(artifacts, val_df: pd.DataFrame, y_val: np.ndarray) -> float:
    features = artifacts.preprocessor.transform(val_df)
    raw_proba = predict_proba(artifacts.model, features)
    cal_proba = apply_calibration(artifacts.calibrator, raw_proba)
    return float(average_precision_score(y_val, cal_proba))


def compute_importance(
    artifacts,
    val_df: pd.DataFrame,
    y_val: np.ndarray,
    baseline_score: float,
) -> dict[str, dict]:
    rng = np.random.RandomState(SEED)
    results: dict[str, dict] = {}
    for feat in ALL_FEATURES:
        drops = []
        for _ in range(N_REPEATS):
            permuted = val_df.copy()
            permuted[feat] = rng.permutation(permuted[feat].values)
            score = _score(artifacts, permuted, y_val)
            drops.append(baseline_score - score)
        results[feat] = {
            "mean_pr_auc_decrease": round(float(np.mean(drops)), 6),
            "std_pr_auc_decrease": round(float(np.std(drops)), 6),
        }
    return results


def make_figure(importance: dict[str, dict]) -> plt.Figure:
    sorted_feats = sorted(
        importance.keys(),
        key=lambda f: importance[f]["mean_pr_auc_decrease"],
    )
    means = [importance[f]["mean_pr_auc_decrease"] for f in sorted_feats]
    labels = [TURKISH_LABELS[f] for f in sorted_feats]

    n = len(sorted_feats)
    dark_teal = np.array([0.0, 0.42, 0.44])
    light_teal = np.array([0.68, 0.85, 0.86])
    accent_orange = np.array([0.90, 0.50, 0.13])

    max_val = max(means) if means else 1.0
    colors = []
    for i, v in enumerate(means):
        if i == n - 1:
            colors.append(accent_orange)
        else:
            t = v / max_val if max_val > 0 else 0
            colors.append(light_teal * (1 - t) + dark_teal * t)

    fig, ax = plt.subplots(figsize=(16, 9), facecolor="white")
    ax.set_facecolor("white")

    bars = ax.barh(range(n), means, color=colors, height=0.65, edgecolor="none")

    for bar, val in zip(bars, means, strict=True):
        offset = max(means) * 0.008 if max(means) > 0 else 0.001
        ax.text(
            bar.get_width() + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            ha="left",
            fontsize=10,
            color="#333333",
        )

    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("PR-AUC düşüşü", fontsize=13, labelpad=10)
    ax.set_title(
        "Global Değişken Önemi",
        fontsize=18,
        fontweight="bold",
        pad=18,
        color="#1a1a1a",
    )
    ax.text(
        0.5,
        1.02,
        "Değişken karıştırıldığında validation PR-AUC değerindeki ortalama düşüş",
        transform=ax.transAxes,
        ha="center",
        fontsize=12,
        color="#555555",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    ax.tick_params(axis="both", colors="#555555")
    ax.xaxis.set_tick_params(labelsize=10)

    x_max = max(means) * 1.18 if means else 1.0
    ax.set_xlim(left=0, right=x_max)

    fig.text(
        0.5,
        0.01,
        "Permütasyon önemi modelin tahmin davranışını özetler; nedensellik göstermez.",
        ha="center",
        fontsize=9,
        color="#888888",
        style="italic",
    )

    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify committed JSON is current.")
    args = parser.parse_args()

    df = load_raw_dataset()
    split = load_split_indices()
    val_df = df.loc[split.validation_index].copy()
    y_val = val_df["Revenue"].astype(int).values
    val_features = val_df[ALL_FEATURES].copy()

    artifacts = load_artifacts()
    baseline = _score(artifacts, val_features, y_val)
    importance = compute_importance(artifacts, val_features, y_val, baseline)

    payload = {
        "baseline_validation_pr_auc": round(baseline, 6),
        "n_repeats": N_REPEATS,
        "seed": SEED,
        "n_features": len(ALL_FEATURES),
        "importance": importance,
    }
    expected = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"

    if args.check:
        if not RESULTS_PATH.exists():
            print(f"FAIL: {RESULTS_PATH.relative_to(PROJECT_ROOT)} does not exist")
            raise SystemExit(1)
        actual = RESULTS_PATH.read_text(encoding="utf-8")
        if actual == expected:
            print("PASS: global feature importance results match recomputation")
        else:
            print("FAIL: results are stale — rerun without --check to regenerate")
            raise SystemExit(1)
        return

    RESULTS_PATH.write_text(expected, encoding="utf-8")
    print(f"Written: {RESULTS_PATH.relative_to(PROJECT_ROOT)}")

    fig = make_figure(importance)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Written: {FIGURE_PATH.relative_to(PROJECT_ROOT)}")

    print(f"\nBaseline validation PR-AUC: {baseline:.6f}")
    print("\nTop 5 most important features:")
    ranked = sorted(importance.items(), key=lambda x: x[1]["mean_pr_auc_decrease"], reverse=True)
    for feat, vals in ranked[:5]:
        print(f"  {feat}: {vals['mean_pr_auc_decrease']:.6f} (±{vals['std_pr_auc_decrease']:.6f})")


if __name__ == "__main__":
    main()
