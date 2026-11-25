"""
Tests for HPC metrics: speedup and parallel efficiency.
"""

import pytest
from src.metrics import calculate_efficiency, calculate_speedup


def test_speedup_calculation():
    # Checks correct speedup ratio S = T_seq / T_par and invalid input rejection
    assert calculate_speedup(10.0, 4.0) == 2.5

    with pytest.raises(ValueError, match="Sequential time"):
        calculate_speedup(0.0, 4.0)
    with pytest.raises(ValueError, match="Sequential time"):
        calculate_speedup(-5.0, 4.0)
    with pytest.raises(ValueError, match="Parallel time"):
        calculate_speedup(10.0, 0.0)
    with pytest.raises(ValueError, match="Parallel time"):
        calculate_speedup(10.0, -2.0)


def test_efficiency_calculation():
    # Checks correct efficiency fraction E = S / p and invalid worker count rejection
    assert calculate_efficiency(2.5, 4) == 0.625

    with pytest.raises(ValueError, match="Worker count"):
        calculate_efficiency(2.5, 0)
    with pytest.raises(ValueError, match="Worker count"):
        calculate_efficiency(2.5, -1)
