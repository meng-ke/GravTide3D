"""Spectral analytic checks that test more than the Poisson trace alone."""
import math

import numpy as np

from gravtide3d import gravitational_potential, tidal_eigensystem
from gravtide3d.physics.poisson import G_CGS


def test_single_fourier_mode_potential():
    shape = (9, 10, 12)
    voxel = (1.1e17, 0.9e17, 1.3e17)
    z, y, x = np.indices(shape)
    phase = 2 * np.pi * (z / shape[0] + 2 * x / shape[2])
    rho_mean, amplitude = 2e-20, 3e-21
    rho = rho_mean + amplitude * np.cos(phase)
    kz = 2 * np.pi / (shape[0] * voxel[0])
    kx = 2 * np.pi * 2 / (shape[2] * voxel[2])
    expected_phi = -4 * math.pi * G_CGS * amplitude * np.cos(phase) / (kz**2 + kx**2)
    actual_phi = gravitational_potential(rho, voxel, memory="in_core").phi
    assert np.max(np.abs(actual_phi - expected_phi)) / np.max(np.abs(expected_phi)) < 1e-12


def test_mixed_fourier_mode_tidal_tensor():
    shape = (9, 10, 12)
    voxel = (1.1e17, 0.9e17, 1.3e17)
    z, y, x = np.indices(shape)
    phase = 2 * np.pi * (z / shape[0] + y / shape[1])
    amp = 4e-21
    rho = 2e-20 + amp * np.cos(phase)
    kz = 2 * np.pi / (shape[0] * voxel[0])
    ky = 2 * np.pi / (shape[1] * voxel[1])
    expected = 4 * math.pi * G_CGS * amp * np.cos(phase) * kz * ky / (kz**2 + ky**2)
    T = tidal_eigensystem(
        rho, voxel, correction="none", return_tensor=True, memory="in_core"
    ).tensor
    assert np.max(np.abs(T[..., 0, 1] - expected)) / np.max(np.abs(expected)) < 1e-12
    assert np.array_equal(T[..., 0, 1], T[..., 1, 0])
    assert np.max(np.abs(T[..., 0, 2])) / np.max(np.abs(expected)) < 1e-12


def test_mixed_nyquist_derivative_is_zero():
    shape = (8, 10, 12)
    z, y, x = np.indices(shape)
    rho = 2e-20 + 1e-21 * np.cos(np.pi * z) * np.cos(2 * np.pi * y / shape[1])
    T = tidal_eigensystem(
        rho, (1e17, 1e17, 1e17), correction="none", return_tensor=True,
        memory="in_core",
    ).tensor
    assert np.max(np.abs(T[..., 0, 1])) < 1e-35
