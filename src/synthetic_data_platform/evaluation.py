from __future__ import annotations

import numpy as np
import pandas as pd


def compare_tables(real: pd.DataFrame, synthetic: pd.DataFrame) -> dict:
    """Report distribution drift and a simple utility-oriented quality score."""
    if list(real.columns) != list(synthetic.columns):
        raise ValueError("Real and synthetic tables must have the same columns in the same order.")
    numeric: dict[str, dict] = {}
    categorical: dict[str, dict] = {}
    component_scores: list[float] = []
    for column in real.columns:
        if pd.api.types.is_numeric_dtype(real[column]):
            real_values = pd.to_numeric(real[column], errors="coerce").dropna().to_numpy(float)
            synthetic_values = pd.to_numeric(synthetic[column], errors="coerce").dropna().to_numpy(float)
            real_std = float(np.std(real_values)) or 1.0
            mean_gap = abs(float(np.mean(real_values)) - float(np.mean(synthetic_values))) / real_std
            std_gap = abs(float(np.std(real_values)) - float(np.std(synthetic_values))) / real_std
            score = max(0.0, 1.0 - min(1.0, 0.5 * mean_gap + 0.5 * std_gap))
            component_scores.append(score)
            numeric[column] = {"real_mean": float(np.mean(real_values)), "synthetic_mean": float(np.mean(synthetic_values)), "real_std": float(np.std(real_values)), "synthetic_std": float(np.std(synthetic_values)), "standardized_mean_gap": mean_gap, "score": score}
        else:
            real_dist = real[column].fillna("__MISSING__").astype(str).value_counts(normalize=True)
            synthetic_dist = synthetic[column].fillna("__MISSING__").astype(str).value_counts(normalize=True)
            categories = set(real_dist.index) | set(synthetic_dist.index)
            total_variation = 0.5 * sum(abs(float(real_dist.get(value, 0.0)) - float(synthetic_dist.get(value, 0.0))) for value in categories)
            score = max(0.0, 1.0 - min(1.0, total_variation))
            component_scores.append(score)
            categorical[column] = {"total_variation_distance": float(total_variation), "score": score}
    return {"rows": {"real": len(real), "synthetic": len(synthetic)}, "numeric": numeric, "categorical": categorical, "quality_score": float(np.mean(component_scores)) if component_scores else 0.0}

