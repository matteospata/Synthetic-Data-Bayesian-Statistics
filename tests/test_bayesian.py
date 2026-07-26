import numpy as np

from synthetic_data_platform.bayesian import normal_inverse_gamma_posterior


def test_bayesian_posterior_is_close_to_observed_mean():
    summary = normal_inverse_gamma_posterior(np.array([9.0, 10.0, 11.0, 10.0]), draws=500, seed=1)
    assert abs(summary["posterior_mean"] - 10.0) < 0.5
    assert summary["credible_interval_95"][0] < summary["posterior_mean"] < summary["credible_interval_95"][1]

