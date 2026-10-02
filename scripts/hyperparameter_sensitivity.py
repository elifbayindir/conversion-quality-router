#!/usr/bin/env python3
"""Bounded validation-only hyperparameter sensitivity study.

Preregistered configurations and seeds are fixed before execution.
Uses only the frozen train/validation splits — the test set is never accessed.
No candidate checkpoints are saved. No frozen artifacts are modified.
"""

import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from conversion_router.data.load import load_raw_dataset
from conversion_router.data.preprocess import load_preprocessor
from conversion_router.data.split import load_split_indices
from conversion_router.data.validate import validate_raw_schema
from conversion_router.modeling.network import predict_proba
from conversion_router.modeling.train import train_mlp

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TARGET_COLUMN = "Revenue"

PREREGISTERED_SEEDS = (17, 42, 73)

PREREGISTERED_CONFIGS: dict[str, dict] = {
    "v1_reference": {
        "hidden_sizes": (128, 64, 32),
        "dropout": 0.30,
        "learning_rate": 0.001,
        "batch_size": 64,
        "max_epochs": 100,
        "patience": 10,
    },
    "lower_dropout": {
        "hidden_sizes": (128, 64, 32),
        "dropout": 0.20,
        "learning_rate": 0.001,
        "batch_size": 64,
        "max_epochs": 100,
        "patience": 10,
    },
    "lower_learning_rate": {
        "hidden_sizes": (128, 64, 32),
        "dropout": 0.30,
        "learning_rate": 0.0003,
        "batch_size": 64,
        "max_epochs": 100,
        "patience": 10,
    },
    "larger_batch": {
        "hidden_sizes": (128, 64, 32),
        "dropout": 0.30,
        "learning_rate": 0.001,
        "batch_size": 128,
        "max_epochs": 100,
        "patience": 10,
    },
    "compact_network": {
        "hidden_sizes": (64, 32),
        "dropout": 0.30,
        "learning_rate": 0.001,
        "batch_size": 64,
        "max_epochs": 100,
        "patience": 10,
    },
}

NOTABLE_CANDIDATE_RULES = {
    "min_pr_auc_improvement": 0.005,
    "min_seed_wins": 2,
    "max_std_increase": 0.005,
    "max_brier_degradation": 0.002,
}

V1_EXPECTED_VAL_PR_AUC_SEED42 = 0.356


@dataclass(frozen=True)
class RunRecord:
    config_name: str
    seed: int
    best_val_pr_auc: float
    val_roc_auc: float
    val_brier_raw: float
    val_log_loss: float
    best_epoch: int
    total_epochs: int
    hidden_sizes: tuple[int, ...]
    dropout: float
    learning_rate: float
    batch_size: int
    max_epochs: int
    patience: int


def evaluate_on_validation(model, X_val: np.ndarray, y_val: np.ndarray) -> dict:
    proba = predict_proba(model, X_val)
    proba_clipped = np.clip(proba, 1e-7, 1.0 - 1e-7)
    return {
        "val_pr_auc": float(average_precision_score(y_val, proba)),
        "val_roc_auc": float(roc_auc_score(y_val, proba)),
        "val_brier_raw": float(brier_score_loss(y_val, proba)),
        "val_log_loss": float(log_loss(y_val, proba_clipped)),
    }


def run_single(
    config_name: str,
    config: dict,
    seed: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> RunRecord:
    result = train_mlp(
        X_train,
        y_train,
        X_val,
        y_val,
        seed=seed,
        max_epochs=config["max_epochs"],
        patience=config["patience"],
        learning_rate=config["learning_rate"],
        batch_size=config["batch_size"],
        hidden_sizes=config["hidden_sizes"],
        dropout=config["dropout"],
    )
    metrics = evaluate_on_validation(result.model, X_val, y_val)
    return RunRecord(
        config_name=config_name,
        seed=seed,
        best_val_pr_auc=metrics["val_pr_auc"],
        val_roc_auc=metrics["val_roc_auc"],
        val_brier_raw=metrics["val_brier_raw"],
        val_log_loss=metrics["val_log_loss"],
        best_epoch=result.best_epoch,
        total_epochs=len(result.history),
        hidden_sizes=config["hidden_sizes"],
        dropout=config["dropout"],
        learning_rate=config["learning_rate"],
        batch_size=config["batch_size"],
        max_epochs=config["max_epochs"],
        patience=config["patience"],
    )


def aggregate_config(records: list[RunRecord]) -> dict:
    pr_aucs = [r.best_val_pr_auc for r in records]
    roc_aucs = [r.val_roc_auc for r in records]
    briers = [r.val_brier_raw for r in records]
    log_losses = [r.val_log_loss for r in records]
    return {
        "mean_val_pr_auc": float(np.mean(pr_aucs)),
        "std_val_pr_auc": float(np.std(pr_aucs, ddof=1)) if len(pr_aucs) > 1 else 0.0,
        "median_val_pr_auc": float(np.median(pr_aucs)),
        "mean_val_roc_auc": float(np.mean(roc_aucs)),
        "mean_val_brier_raw": float(np.mean(briers)),
        "mean_val_log_loss": float(np.mean(log_losses)),
    }


def check_notable_candidate(
    config_agg: dict,
    ref_agg: dict,
    config_records: list[RunRecord],
    ref_records: list[RunRecord],
) -> dict:
    rules = NOTABLE_CANDIDATE_RULES
    pr_auc_improvement = config_agg["mean_val_pr_auc"] - ref_agg["mean_val_pr_auc"]
    std_increase = config_agg["std_val_pr_auc"] - ref_agg["std_val_pr_auc"]
    brier_degradation = config_agg["mean_val_brier_raw"] - ref_agg["mean_val_brier_raw"]

    seed_wins = sum(
        1
        for cr, rr in zip(
            sorted(config_records, key=lambda r: r.seed),
            sorted(ref_records, key=lambda r: r.seed),
            strict=True,
        )
        if cr.best_val_pr_auc > rr.best_val_pr_auc
    )

    passes = {
        "pr_auc_improvement_ge_0.005": pr_auc_improvement >= rules["min_pr_auc_improvement"],
        "seed_wins_ge_2_of_3": seed_wins >= rules["min_seed_wins"],
        "std_increase_le_0.005": std_increase <= rules["max_std_increase"],
        "brier_degradation_le_0.002": brier_degradation <= rules["max_brier_degradation"],
    }

    return {
        "pr_auc_improvement": round(pr_auc_improvement, 6),
        "seed_wins": seed_wins,
        "std_increase": round(std_increase, 6),
        "brier_degradation": round(brier_degradation, 6),
        "criteria": passes,
        "meets_all": all(passes.values()),
    }


def main() -> dict:
    print("Loading data and preprocessor...")
    df = load_raw_dataset()
    validate_raw_schema(df)
    split = load_split_indices()

    train_df = df.loc[split.train_index]
    val_df = df.loc[split.validation_index]

    preprocessor = load_preprocessor()
    X_train = preprocessor.transform(train_df)
    X_val = preprocessor.transform(val_df)
    y_train = train_df[TARGET_COLUMN].to_numpy().astype(int)
    y_val = val_df[TARGET_COLUMN].to_numpy().astype(int)

    print(f"Train: {X_train.shape[0]} rows, Val: {X_val.shape[0]} rows")
    print(f"Configurations: {len(PREREGISTERED_CONFIGS)}, Seeds: {len(PREREGISTERED_SEEDS)}")
    print(f"Total runs: {len(PREREGISTERED_CONFIGS) * len(PREREGISTERED_SEEDS)}")

    print("\n--- Reference reproducibility guard ---")
    ref_config = PREREGISTERED_CONFIGS["v1_reference"]
    ref_guard = run_single(
        "v1_reference", ref_config, 42, X_train, y_train, X_val, y_val,
    )
    print(f"v1_reference seed=42: val PR-AUC = {ref_guard.best_val_pr_auc:.4f}")
    print(f"Expected: ~{V1_EXPECTED_VAL_PR_AUC_SEED42}")
    if abs(ref_guard.best_val_pr_auc - V1_EXPECTED_VAL_PR_AUC_SEED42) > 0.002:
        print("MISMATCH: material difference from frozen v1 record. Stopping.")
        sys.exit(1)
    print("Reproducibility confirmed.\n")

    print("--- Running full study ---")
    all_records: list[RunRecord] = []
    for config_name, config in PREREGISTERED_CONFIGS.items():
        for seed in PREREGISTERED_SEEDS:
            if config_name == "v1_reference" and seed == 42:
                all_records.append(ref_guard)
                print(f"  {config_name} seed={seed}: {ref_guard.best_val_pr_auc:.4f} (reused)")
                continue
            record = run_single(config_name, config, seed, X_train, y_train, X_val, y_val)
            all_records.append(record)
            print(f"  {config_name} seed={seed}: {record.best_val_pr_auc:.4f}")

    print(f"\nCompleted {len(all_records)} runs.")

    records_by_config: dict[str, list[RunRecord]] = {}
    for r in all_records:
        records_by_config.setdefault(r.config_name, []).append(r)

    aggregates: dict[str, dict] = {}
    for name, recs in records_by_config.items():
        aggregates[name] = aggregate_config(recs)

    ref_agg = aggregates["v1_reference"]
    ref_records = records_by_config["v1_reference"]

    candidate_evaluations: dict[str, dict] = {}
    for name in PREREGISTERED_CONFIGS:
        if name == "v1_reference":
            continue
        candidate_evaluations[name] = check_notable_candidate(
            aggregates[name], ref_agg, records_by_config[name], ref_records
        )

    notable_candidates = [
        name for name, ev in candidate_evaluations.items() if ev["meets_all"]
    ]

    best_candidate = None
    if notable_candidates:
        best_candidate = max(
            notable_candidates, key=lambda n: aggregates[n]["mean_val_pr_auc"]
        )

    results = {
        "study_metadata": {
            "type": "bounded_validation_only_hyperparameter_sensitivity",
            "preregistered": True,
            "n_configurations": len(PREREGISTERED_CONFIGS),
            "n_seeds": len(PREREGISTERED_SEEDS),
            "total_runs": len(all_records),
            "seeds": list(PREREGISTERED_SEEDS),
            "configurations": {
                name: {
                    "hidden_sizes": list(cfg["hidden_sizes"]),
                    "dropout": cfg["dropout"],
                    "learning_rate": cfg["learning_rate"],
                    "batch_size": cfg["batch_size"],
                    "max_epochs": cfg["max_epochs"],
                    "patience": cfg["patience"],
                }
                for name, cfg in PREREGISTERED_CONFIGS.items()
            },
            "notable_candidate_rules": NOTABLE_CANDIDATE_RULES,
            "data_protocol": "train_and_validation_only",
            "test_set_accessed": False,
            "candidate_checkpoints_saved": False,
            "v1_remains_authoritative": True,
        },
        "reference_reproducibility": {
            "seed": 42,
            "observed_val_pr_auc": round(ref_guard.best_val_pr_auc, 4),
            "expected_val_pr_auc": V1_EXPECTED_VAL_PR_AUC_SEED42,
            "match": abs(ref_guard.best_val_pr_auc - V1_EXPECTED_VAL_PR_AUC_SEED42) <= 0.002,
        },
        "per_run_results": [
            {
                "config_name": r.config_name,
                "seed": r.seed,
                "best_val_pr_auc": round(r.best_val_pr_auc, 4),
                "val_roc_auc": round(r.val_roc_auc, 4),
                "val_brier_raw": round(r.val_brier_raw, 4),
                "val_log_loss": round(r.val_log_loss, 4),
                "best_epoch": r.best_epoch,
                "total_epochs": r.total_epochs,
            }
            for r in all_records
        ],
        "aggregate_results": {
            name: {k: round(v, 4) for k, v in agg.items()}
            for name, agg in aggregates.items()
        },
        "paired_comparisons": {
            name: {
                "per_seed": [
                    {
                        "seed": cr.seed,
                        "config_pr_auc": round(cr.best_val_pr_auc, 4),
                        "reference_pr_auc": round(rr.best_val_pr_auc, 4),
                        "delta": round(cr.best_val_pr_auc - rr.best_val_pr_auc, 4),
                    }
                    for cr, rr in zip(
                        sorted(records_by_config[name], key=lambda r: r.seed),
                        sorted(ref_records, key=lambda r: r.seed),
                        strict=True,
                    )
                ],
                **{k: round(v, 4) if isinstance(v, float) else v for k, v in ev.items()},
            }
            for name, ev in candidate_evaluations.items()
        },
        "candidate_outcome": {
            "notable_candidates": notable_candidates,
            "best_candidate": best_candidate,
            "label": f"candidate_v2 ({best_candidate})" if best_candidate else None,
            "status": "validation_evidence_only",
            "promoted": False,
            "v1_authoritative": True,
            "note": (
                "No robust validation-only candidate met the "
                "preregistered threshold."
                if not best_candidate
                else (
                    f"Configuration '{best_candidate}' met all "
                    "preregistered criteria on validation data only. "
                    "This is not a promotion. V1 remains authoritative. "
                    "Calibration-sensitive promotion would require a "
                    "separate controlled protocol."
                )
            ),
        },
        "package_versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }

    out_path = PROJECT_ROOT / "docs" / "hyperparameter_sensitivity_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out_path}")

    print("\n--- Configuration ranking by mean validation PR-AUC ---")
    ranked = sorted(aggregates.items(), key=lambda x: x[1]["mean_val_pr_auc"], reverse=True)
    for i, (name, agg) in enumerate(ranked, 1):
        marker = " *" if name == best_candidate else ""
        print(
            f"  {i}. {name}: mean={agg['mean_val_pr_auc']:.4f} "
            f"std={agg['std_val_pr_auc']:.4f} "
            f"brier={agg['mean_val_brier_raw']:.4f}{marker}"
        )

    if best_candidate:
        print(f"\nValidation candidate: candidate_v2 ({best_candidate})")
        print("This is validation evidence only. V1 remains authoritative.")
    else:
        print("\nNo robust validation-only candidate met the preregistered threshold.")

    return results


if __name__ == "__main__":
    main()
