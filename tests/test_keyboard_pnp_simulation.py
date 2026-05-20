import pytest

from sim_keyboard_pnp import run_simulation


def test_calculated_simulation_recovers_key_from_sparse_accumulated_frames():
    result = run_simulation(
        target_key="a",
        frames=5,
        keys_per_frame=3,
        noise_px=0.0,
        seed=7,
    )

    assert max(result.layout_keys_per_frame) <= 3
    assert result.unique_layout_keys >= 4
    assert result.reproj_err_px < 1e-6
    assert result.target_error_mm < 1e-5


def test_calculated_simulation_single_sparse_frame_fails_without_cache():
    with pytest.raises(ValueError, match="Only 3 detected keys"):
        run_simulation(
            target_key="a",
            frames=1,
            keys_per_frame=3,
            noise_px=0.0,
            seed=7,
        )


def test_calculated_simulation_handles_noisy_sparse_detections():
    result = run_simulation(
        target_key="enter",
        frames=5,
        keys_per_frame=4,
        noise_px=1.5,
        duplicate_rate=0.5,
        seed=11,
    )

    assert max(result.layout_keys_per_frame) <= 4
    assert result.unique_layout_keys >= 4
    assert result.reproj_err_px < 4.0
    assert result.target_error_mm < 20.0
