import numpy as np
import pytest

from gravtide3d.validation.inputs import validate_density


def test_rejects_non_3d():
    with pytest.raises(ValueError):
        validate_density(np.zeros((4, 5)))


def test_rejects_negative_density():
    rho = np.ones((3, 4, 5))
    rho[1, 2, 3] = -1
    with pytest.raises(ValueError):
        validate_density(rho)
