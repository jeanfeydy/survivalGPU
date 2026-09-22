"""Regression test for the pykeops-optional guarding mechanism.

Uses `monkeypatch` to force `survivalgpu.wce_features.KEOPS_AVAILABLE` to
False for the duration of a single test, without requiring pykeops to
actually be absent from the environment. These tests must run
unconditionally (not marked `needs_keops`): they test the guard itself,
not the WCE model's numerics.
"""

import numpy as np
import pytest
from survivalgpu import (
    CoxPHSurvivalAnalysis,
    WCESurvivalAnalysis,
    wce_features,
    wce_numpy,
)
from survivalgpu.datasets import simple_dataset


def _toy_wce_data():
    """A small, deterministic long-format dataset: 2 patients, 5 daily intervals each.

    WCESurvivalAnalysis/wce_numpy expect one row per (patient, day), with
    `dose` set to 0 on days without an intake -- unlike `load_drugs()`,
    whose `dose`/`dose_time`/... arrays only list the sparse drug events
    and don't align 1:1 with `start`/`stop`/`event`/`patient`.
    """
    patient = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int64)
    start = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], dtype=np.int64)
    stop = start + 1
    dose = np.array(
        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64
    )
    event = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=np.int64)
    return dict(
        dose=dose, start=start, stop=stop, event=event, patient=patient
    )


def test_coxph_still_works_without_keops(monkeypatch):
    """CoxPH must keep working even when pykeops is unavailable."""
    monkeypatch.setattr(wce_features, "KEOPS_AVAILABLE", False)

    ds = simple_dataset(
        n_covariates=2,
        n_patients=30,
        max_duration=5,
        ensure_one_life=True,
        ensure_one_death=True,
        unit_length_intervals=True,
    )
    model = CoxPHSurvivalAnalysis(ties="breslow")
    model.fit(
        covariates=ds.covariates,
        start=ds.start,
        stop=ds.stop,
        event=ds.event,
        batch=ds.batch,
        strata=ds.strata,
    )
    assert model.coef_.shape == (1, 2)


def test_wce_class_raises_clear_error_without_keops(monkeypatch):
    """WCESurvivalAnalysis.fit() must raise a clear error mentioning pykeops."""
    monkeypatch.setattr(wce_features, "KEOPS_AVAILABLE", False)

    data = _toy_wce_data()
    model = WCESurvivalAnalysis(cutoff=5, order=3, nknots=1)
    with pytest.raises(ImportError, match="pykeops"):
        model.fit(**data)


def test_wce_numpy_raises_clear_error_without_keops(monkeypatch):
    """wce_numpy must raise the same clear error mentioning pykeops."""
    monkeypatch.setattr(wce_features, "KEOPS_AVAILABLE", False)

    data = _toy_wce_data()
    with pytest.raises(ImportError, match="pykeops"):
        wce_numpy(
            ids=data["patient"],
            covariates=None,
            doses=data["dose"],
            events=data["event"],
            start=data["start"],
            stop=data["stop"],
            cutoff=5,
        )
