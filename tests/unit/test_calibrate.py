import numpy as np
import pytest

from conversion_router.modeling.calibrate import (
    UNCERTAINTY_BAND,
    apply_calibration,
    calibration_evidence,
    decision_margin,
    fit_calibrator,
    is_uncertain,
    load_calibrator,
    save_calibrator,
)


def _miscalibrated_data(n: int = 800, seed: int = 0):
    """Raw scores that are systematically overconfident vs. true labels."""
    rng = np.random.default_rng(seed)
    true_prob = rng.uniform(0, 1, size=n)
    y_true = (rng.random(n) < true_prob).astype(int)
    # Overconfident: push raw scores toward the extremes vs. the true probability.
    raw_proba = np.clip(true_prob + 0.3 * np.sign(true_prob - 0.5) * rng.random(n), 0, 1)
    return y_true, raw_proba


def test_calibration_improves_or_maintains_brier_score():
    y_true, raw_proba = _miscalibrated_data()
    calibrator = fit_calibrator(y_true, raw_proba)
    calibrated_proba = apply_calibration(calibrator, raw_proba)

    evidence = calibration_evidence(y_true, raw_proba, calibrated_proba)

    assert evidence.brier_score_calibrated <= evidence.brier_score_raw + 1e-9


def test_calibrated_probabilities_are_in_unit_interval():
    y_true, raw_proba = _miscalibrated_data()
    calibrator = fit_calibrator(y_true, raw_proba)
    calibrated_proba = apply_calibration(calibrator, raw_proba)

    assert (calibrated_proba >= 0).all()
    assert (calibrated_proba <= 1).all()


def test_calibration_preserves_rank_order_isotonic():
    y_true, raw_proba = _miscalibrated_data()
    calibrator = fit_calibrator(y_true, raw_proba)

    sample = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    calibrated_sample = apply_calibration(calibrator, sample)

    assert np.all(np.diff(calibrated_sample) >= -1e-9)  # non-decreasing


def test_save_and_load_calibrator_roundtrip(tmp_path):
    y_true, raw_proba = _miscalibrated_data()
    calibrator = fit_calibrator(y_true, raw_proba)
    expected = apply_calibration(calibrator, raw_proba)

    path = tmp_path / "calibrator.joblib"
    save_calibrator(calibrator, path)
    reloaded = load_calibrator(path)
    actual = apply_calibration(reloaded, raw_proba)

    assert np.allclose(expected, actual)


def test_load_missing_calibrator_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_calibrator(tmp_path / "does_not_exist.joblib")


def test_decision_margin_arithmetic():
    margins = decision_margin([0.3, 0.5, 0.9], threshold=0.5)
    assert np.allclose(margins, [0.2, 0.0, 0.4])


def test_is_uncertain_inside_and_outside_the_band():
    threshold = 0.5
    band = 0.05
    proba = [0.40, 0.46, 0.50, 0.54, 0.60]
    # margins:  0.10   0.04   0.00   0.04   0.10

    flags = is_uncertain(proba, threshold, band=band)

    assert list(flags) == [False, True, True, True, False]


def test_default_uncertainty_band_is_versioned_constant():
    assert UNCERTAINTY_BAND == 0.05
