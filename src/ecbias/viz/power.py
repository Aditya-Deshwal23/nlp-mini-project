"""Seeded power simulation used by the offline dashboard."""

from __future__ import annotations

import numpy as np
from scipy.stats import ttest_ind


def simulate_power(
    effect_size: float,
    n_female: int,
    n_male: int,
    n_studies: int,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[np.ndarray, float]:
    """Simulate two-sided standardised mean differences and return p-values/power."""
    rng = np.random.default_rng(seed)
    p_values = np.empty(n_studies)
    for index in range(n_studies):
        female = rng.normal(effect_size, 1, n_female)
        male = rng.normal(0, 1, n_male)
        p_values[index] = ttest_ind(female, male, equal_var=False).pvalue
    return p_values, float(np.mean(p_values < alpha))
