from __future__ import annotations

import numpy as np

from backscatter.iterative import (
    ReconstructionGrid2D,
    denoise_tv_chambolle,
    listmode_mlem,
)
from scipy.sparse import csr_matrix


def test_grid_shape_and_centers() -> None:
    grid = ReconstructionGrid2D(
        x_edges_mm=np.array([-2.0, 0.0, 2.0]),
        depth_edges_mm=np.array([0.0, 2.0, 4.0, 6.0]),
    )
    assert grid.shape == (3, 2)
    assert grid.n_pixels == 6
    assert np.allclose(grid.x_centers_mm, [-1.0, 1.0])
    assert np.allclose(grid.depth_centers_mm, [1.0, 3.0, 5.0])


def test_listmode_mlem_identity_model_recovers_event_rate() -> None:
    matrix = csr_matrix(np.eye(3))
    sensitivity = np.ones(3) / 10.0
    image, history = listmode_mlem(
        matrix,
        sensitivity,
        target_exposure=10.0,
        iterations=2,
    )
    assert np.allclose(image, 1.0)
    assert len(history) == 2


def test_tv_denoising_preserves_constant_and_reduces_impulse() -> None:
    constant = np.full((12, 14), 2.5)
    assert np.allclose(
        denoise_tv_chambolle(constant, weight=0.1),
        constant,
    )
    noisy = np.zeros((12, 14))
    noisy[6, 7] = 1.0
    filtered = denoise_tv_chambolle(noisy, weight=0.1)
    assert 0.0 < filtered[6, 7] < 1.0
    assert np.std(filtered) < np.std(noisy)

