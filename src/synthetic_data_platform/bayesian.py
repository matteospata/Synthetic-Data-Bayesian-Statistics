from __future__ import annotations

import numpy as np
import pandas as pd


def normal_inverse_gamma_posterior(values: np.ndarray, draws: int = 2000, seed: int = 42) -> dict:
    """Sample a posterior for a normal mean and variance with a weak prior."""
    observations = np.asarray(values, dtype=float)
    observations = observations[np.isfinite(observations)]
    if len(observations) < 2:
        raise ValueError("At least two numeric observations are required.")
    mu_0, kappa_0, alpha_0, beta_0 = float(np.mean(observations)), 1.0, 2.0, 1.0
    n, sample_mean = len(observations), float(np.mean(observations))
    sum_squares = float(np.sum((observations - sample_mean) ** 2))
    kappa_n = kappa_0 + n
    mu_n = (kappa_0 * mu_0 + n * sample_mean) / kappa_n
    alpha_n = alpha_0 + n / 2.0
    beta_n = beta_0 + 0.5 * sum_squares + (kappa_0 * n * (sample_mean - mu_0) ** 2) / (2.0 * kappa_n)
    rng = np.random.default_rng(seed)
    variance = 1.0 / rng.gamma(shape=alpha_n, scale=1.0 / beta_n, size=draws)
    means = rng.normal(loc=mu_n, scale=np.sqrt(variance / kappa_n))
    return {"n": n, "posterior_mean": float(np.mean(means)), "credible_interval_95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))], "posterior_std": float(np.std(means)), "prior": {"mu_0": mu_0, "kappa_0": kappa_0, "alpha_0": alpha_0, "beta_0": beta_0}}


def compare_posteriors(real: pd.DataFrame, synthetic: pd.DataFrame, draws: int = 2000, seed: int = 42) -> dict:
    numeric_columns = [column for column in real.columns if pd.api.types.is_numeric_dtype(real[column]) and pd.api.types.is_numeric_dtype(synthetic[column])]
    report: dict[str, dict] = {}
    for offset, column in enumerate(numeric_columns):
        real_summary = normal_inverse_gamma_posterior(real[column].to_numpy(), draws, seed + offset)
        synthetic_summary = normal_inverse_gamma_posterior(synthetic[column].to_numpy(), draws, seed + 1000 + offset)
        rng = np.random.default_rng(seed + 2000 + offset)
        real_draws = rng.normal(real_summary["posterior_mean"], real_summary["posterior_std"], draws)
        synthetic_draws = rng.normal(synthetic_summary["posterior_mean"], synthetic_summary["posterior_std"], draws)
        report[column] = {"real": real_summary, "synthetic": synthetic_summary, "posterior_mean_difference": real_summary["posterior_mean"] - synthetic_summary["posterior_mean"], "probability_real_mean_greater": float(np.mean(real_draws > synthetic_draws))}
    return {"method": "Normal-Inverse-Gamma conjugate posterior", "columns": report}

