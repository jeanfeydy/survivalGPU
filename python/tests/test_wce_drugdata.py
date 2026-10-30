import numpy as np
import pytest
import rpy2.robjects as ro
import rpy2.robjects.packages as rpackages
import torch
from rpy2.robjects import conversion, pandas2ri
from survivalgpu import WCESurvivalAnalysis
from survivalgpu.wce import wce_numpy, wce_R

# rpy2 + R's WCE package are a pre-existing hard dependency of the `test`
# dependency-group, unrelated to pykeops -- kept at module level.
WCE = rpackages.importr("WCE")
ro.r("data('drugdata')")
drugdata_r = ro.r["drugdata"]


conv = conversion.get_conversion()
with conversion.localconverter(conv + pandas2ri.converter):
    drugdata_df = pandas2ri.rpy2py(drugdata_r)

_dose = np.array(drugdata_df["dose"], dtype=np.float64)
_stop = np.array(drugdata_df["Stop"], dtype=np.int64)
_start = np.array(drugdata_df["Start"], dtype=np.int64)
_patient = np.array(drugdata_df["Id"], dtype=np.int64)
_event = np.array(drugdata_df["Event"], dtype=np.int64)


@pytest.mark.needs_keops()
@pytest.mark.parametrize("nknots", [1, 2, 3])
@pytest.mark.parametrize("cutoff", [90, 180])
def test_wce_drugdata(cutoff, nknots):
    """Fits WCESurvivalAnalysis on the reference WCE R package's `drugdata`.

    `cutoff=90` matches the value used in the R vignettes (WCE.Rmd,
    survivalGPU.Rmd); `cutoff=180` matches the Python examples
    (03_wce_basic.py, 04_wce_bootstrap.py).
    """
    model = WCESurvivalAnalysis(
        cutoff=cutoff, constrained="right", nknots=nknots
    )

    model.fit(
        dose=_dose,
        stop=_stop,
        start=_start,
        patient=_patient,
        event=_event,
    )

    assert model.WCE_coef_.shape == (1, model.n_atoms)
    assert model.risk_function_.shape == (1, cutoff)
    assert np.all(np.isfinite(model.coef_))
    assert np.all(np.isfinite(model.WCE_coef_))


@pytest.mark.needs_keops()
def test_wce_drugdata_bootstrap():
    """Characterizes WCESurvivalAnalysis's existing bootstrap support.

    This combination (nbootstraps set on a WCE model) has no prior test coverage
    anywhere in the suite, so this is a safety net ahead of upcoming changes that
    let the bootstrap select the best nknots per replicate from a candidate list.
    """
    cutoff = 90
    nbootstraps = 20
    model = WCESurvivalAnalysis(
        cutoff=cutoff,
        constrained="right",
        nknots=2,
        nbootstraps=nbootstraps,
        batchsize=10,
    )

    model.fit(
        dose=_dose,
        stop=_stop,
        start=_start,
        patient=_patient,
        event=_event,
    )

    assert model.bootstrap_coef_.shape == (nbootstraps, 1, model.n_covariates)
    assert model.bootstrap_WCE_coef_.shape == (nbootstraps, 1, model.n_atoms)
    assert model.bootstrap_risk_functions_.shape == (nbootstraps, 1, cutoff)
    assert np.all(np.isfinite(model.bootstrap_coef_))
    assert np.all(np.isfinite(model.bootstrap_WCE_coef_))


def test_wce_n_atoms_for_is_parametrized():
    """`_n_atoms_for` must depend on its `nknots` argument, not on `self.nknots`.

    Regression test for the refactor that turns `n_atoms` into a thin wrapper
    around `_n_atoms_for(self.nknots)`, ahead of `nknots` becoming a list of
    candidates: proves the parameter is really used, not silently ignored.
    """
    model = WCESurvivalAnalysis(cutoff=90, nknots=1)
    assert model.n_atoms == 1 + model.order + 1
    assert model._n_atoms_for(3) == 3 + model.order + 1
    assert model.n_atoms == 1 + model.order + 1


def test_wce_nknots_type_is_checked():
    """nknots accepts an int or a list/tuple of ints, and rejects other types.

    `nknots` is annotated `Int | list[Int] | tuple[Int, ...]` on `__init__`,
    the same pattern used for `constrained: Literal[...] | None`: a fast,
    signature-level type gate on top of the exhaustive value checks already
    done in the `nknots` setter.
    """
    assert WCESurvivalAnalysis(cutoff=90, nknots=3).nknots == 3

    with pytest.raises(TypeError):
        WCESurvivalAnalysis(cutoff=90, nknots="3")

    with pytest.raises(TypeError):
        WCESurvivalAnalysis(cutoff=90, nknots=[1, "2", 3])


def test_wce_nknots_candidates_sorted_and_deduplicated():
    """A list of candidate nknots is stored sorted and without duplicates.

    Mirrors the reference WCE R package's own `nknots <- sort(unique(nknots))`.
    """
    model = WCESurvivalAnalysis(cutoff=90, nknots=[3, 1, 2, 1])
    assert model.nknots_candidates == (1, 2, 3)


@pytest.mark.needs_keops()
def test_wce_drugdata_multi_knot_grid_shapes():
    """Fitting several candidate nknots values populates the `*_grid_` attributes."""
    cutoff = 90
    candidates = (1, 2, 3)
    model = WCESurvivalAnalysis(
        cutoff=cutoff, constrained="right", nknots=list(candidates)
    )

    model.fit(
        dose=_dose,
        stop=_stop,
        start=_start,
        patient=_patient,
        event=_event,
    )

    n_candidates = len(candidates)
    assert model.nknots_candidates == candidates
    assert model.info_criterion_grid_.shape == (n_candidates, 1)
    assert model.loglik_grid_.shape == (n_candidates, 1)
    assert model.coef_grid_.shape == (n_candidates, 1, model.n_covariates)
    assert model.std_grid_.shape == (n_candidates, 1, model.n_covariates)
    assert model.risk_function_grid_.shape == (n_candidates, 1, cutoff)
    assert len(model.knots_grid_) == n_candidates
    assert len(model.WCE_coef_grid_) == n_candidates
    assert len(model.SED_grid_) == n_candidates
    assert model.best_index_.shape == (1,)
    assert model.best_nknots_.shape == (1,)
    assert model.best_nknots_[0] in candidates
    assert np.all(np.isfinite(model.info_criterion_grid_))

    # Regression: with a single batch column (always true today, since
    # `batch` is never used), WCE_coef_/knots_/SED_ must stay dense arrays,
    # not a length-1 list -- a list here would convert to an R *list* (not a
    # matrix/vector) via reticulate, breaking wceGPU()'s ncol()/rownames()
    # calls even though there is no genuine per-batch raggedness to justify it.
    assert isinstance(model.WCE_coef_, np.ndarray)
    assert isinstance(model.SED_, np.ndarray)
    assert isinstance(model.knots_, np.ndarray)
    assert model.WCE_coef_.shape == (
        1,
        model._n_atoms_for(model.best_nknots_[0]),
    )

    # Regression: "usual CoxPH results" (imat_ in particular) must reflect
    # the *selected* candidate, not simply whichever one was fitted last in
    # the loop -- otherwise imat_'s dimensions (sized by the last candidate's
    # n_atoms) would mismatch coef_/WCE_coef_'s (sized by the winner's).
    n_total = model.n_covariates + model._n_atoms_for(model.best_nknots_[0])
    assert model.imat_.shape == (1, n_total, n_total)
    assert model.means_.shape == (n_total,)


@pytest.mark.needs_keops()
def test_wce_drugdata_multi_knot_matches_independent_scalar_fits():
    """Each candidate in the grid reproduces its own independent scalar fit."""
    cutoff = 90
    candidates = (1, 2, 3)

    grid_model = WCESurvivalAnalysis(
        cutoff=cutoff, constrained="right", nknots=list(candidates)
    )
    grid_model.fit(
        dose=_dose, stop=_stop, start=_start, patient=_patient, event=_event
    )

    for i, nknots in enumerate(candidates):
        scalar_model = WCESurvivalAnalysis(
            cutoff=cutoff, constrained="right", nknots=nknots
        )
        scalar_model.fit(
            dose=_dose,
            stop=_stop,
            start=_start,
            patient=_patient,
            event=_event,
        )
        assert np.allclose(
            grid_model.info_criterion_grid_[i, 0],
            scalar_model.info_criterion_[0],
        )
        assert np.allclose(
            grid_model.loglik_grid_[i, 0], scalar_model.loglik_[0]
        )

    assert (
        grid_model.info_criterion_[0]
        == grid_model.info_criterion_grid_[:, 0].min()
    )


@pytest.mark.needs_keops()
def test_wce_drugdata_single_candidate_list_matches_scalar():
    """`nknots=2` and `nknots=[2]` (a grid of one) must be numerically identical."""
    cutoff = 90

    scalar_model = WCESurvivalAnalysis(
        cutoff=cutoff, constrained="right", nknots=2
    )
    scalar_model.fit(
        dose=_dose, stop=_stop, start=_start, patient=_patient, event=_event
    )

    list_model = WCESurvivalAnalysis(
        cutoff=cutoff, constrained="right", nknots=[2]
    )
    list_model.fit(
        dose=_dose, stop=_stop, start=_start, patient=_patient, event=_event
    )

    assert np.allclose(scalar_model.coef_, list_model.coef_)
    assert np.allclose(scalar_model.WCE_coef_, list_model.WCE_coef_)
    assert np.allclose(
        scalar_model.risk_function_.cpu().numpy(),
        list_model.risk_function_.cpu().numpy(),
    )
    assert np.allclose(
        scalar_model.info_criterion_, list_model.info_criterion_
    )


@pytest.mark.needs_keops()
def test_wce_drugdata_multi_knot_bootstrap_grid_shapes():
    """Combining several candidate nknots with bootstrapping populates the bootstrap grid.

    Each candidate reuses the same resample plan -- checked via the
    `bootstrap_n_events_` invariant asserted inside `.fit()` itself, which
    would fail loudly if resamples were ever redrawn per candidate instead
    of shared.
    """
    cutoff = 90
    candidates = (1, 2, 3)
    nbootstraps = 8
    model = WCESurvivalAnalysis(
        cutoff=cutoff,
        constrained="right",
        nknots=list(candidates),
        nbootstraps=nbootstraps,
        batchsize=4,
    )

    model.fit(
        dose=_dose,
        stop=_stop,
        start=_start,
        patient=_patient,
        event=_event,
    )

    n_candidates = len(candidates)
    assert model.bootstrap_coef_grid_.shape == (
        n_candidates,
        nbootstraps,
        1,
        model.n_covariates,
    )
    assert model.bootstrap_risk_functions_grid_.shape == (
        n_candidates,
        nbootstraps,
        1,
        cutoff,
    )
    assert model.bootstrap_loglik_grid_.shape == (n_candidates, nbootstraps, 1)
    assert len(model.bootstrap_WCE_coef_grid_) == n_candidates
    assert model.bootstrap_n_events_.shape == (nbootstraps,)


@pytest.mark.needs_keops()
def test_wce_drugdata_bootstrap_selects_best_knot_per_replicate():
    """Each bootstrap replicate independently selects its own best nknots.

    This is the actual feature: bootstrap_coef_/bootstrap_risk_functions_
    reflect, for every replicate, whichever candidate minimized the
    information criterion on that replicate's own resampled data.
    """
    cutoff = 90
    candidates = (1, 2, 3)
    nbootstraps = 20
    model = WCESurvivalAnalysis(
        cutoff=cutoff,
        constrained="right",
        nknots=list(candidates),
        nbootstraps=nbootstraps,
        batchsize=10,
    )

    model.fit(
        dose=_dose,
        stop=_stop,
        start=_start,
        patient=_patient,
        event=_event,
    )

    assert model.bootstrap_best_index_.shape == (nbootstraps, 1)
    assert model.bootstrap_best_nknots_.shape == (nbootstraps, 1)
    assert set(np.unique(model.bootstrap_best_nknots_)).issubset(
        set(candidates)
    )
    assert model.bootstrap_risk_functions_.shape == (nbootstraps, 1, cutoff)
    assert model.bootstrap_coef_.shape == (nbootstraps, 1, model.n_covariates)

    # Argmin correctness: the selected candidate's own criterion must equal
    # the row-wise minimum across all candidates, for every replicate.
    for r in range(nbootstraps):
        selected = model.bootstrap_best_index_[r, 0]
        assert (
            model.bootstrap_info_criterion_grid_[selected, r, 0]
            == model.bootstrap_info_criterion_grid_[:, r, 0].min()
        )


@pytest.mark.needs_keops()
def test_wce_drugdata_single_candidate_list_matches_scalar_with_bootstrap():
    """`nknots=[2]` bootstrap output must match `nknots=2`'s, byte for byte.

    Proves the grid-of-one bootstrap path (Commits 5a/5b) is identical to
    today's scalar-nknots bootstrap path (Commit 0), just routed through the
    per-replicate selection machinery with a single candidate to "select".
    """
    cutoff = 90
    nbootstraps = 10

    torch.manual_seed(0)
    scalar_model = WCESurvivalAnalysis(
        cutoff=cutoff,
        constrained="right",
        nknots=2,
        nbootstraps=nbootstraps,
        batchsize=5,
    )
    scalar_model.fit(
        dose=_dose, stop=_stop, start=_start, patient=_patient, event=_event
    )

    torch.manual_seed(0)
    list_model = WCESurvivalAnalysis(
        cutoff=cutoff,
        constrained="right",
        nknots=[2],
        nbootstraps=nbootstraps,
        batchsize=5,
    )
    list_model.fit(
        dose=_dose, stop=_stop, start=_start, patient=_patient, event=_event
    )

    assert np.array_equal(
        scalar_model.bootstrap_coef_, list_model.bootstrap_coef_
    )
    assert torch.equal(
        scalar_model.bootstrap_risk_functions_,
        list_model.bootstrap_risk_functions_,
    )


@pytest.mark.needs_keops()
def test_wce_numpy_accepts_nknots_list():
    """wce_numpy (the R bridge's computational backend) forwards a list nknots.

    Also checks the new diagnostic keys it exposes for the R port: best_nknots,
    nknots_grid, info_criterion_grid, loglik_grid.
    """
    res = wce_numpy(
        ids=_patient,
        covariates=None,
        doses=_dose,
        events=_event,
        start=_start,
        stop=_stop,
        cutoff=90,
        nknots=[1, 2, 3],
        constrained="right",
    )

    assert res["nknots_grid"].tolist() == [1, 2, 3]
    assert res["info_criterion_grid"].shape == (3, 1)
    assert res["loglik_grid"].shape == (3, 1)
    assert res["best_nknots"].shape == (1,)
    assert res["best_nknots"][0] in (1, 2, 3)
    assert res["info_criterion"] == res["info_criterion_grid"].min()


@pytest.mark.needs_keops()
def test_wce_r_accepts_nknots_vector():
    """wce_R (the actual R-facing entry point) accepts a list or a NumPy array
    for nknots -- covering both a plain Python list and whatever reticulate is
    likely to hand over for an R numeric vector -- as well as a plain scalar.
    """
    common_kwargs = dict(
        data=drugdata_df,
        ids="Id",
        covars=None,
        start="Start",
        stop="Stop",
        doses="dose",
        events="Event",
        cutoff=90,
        constrained="R",
        aic=False,
        bootstrap=0,
        batchsize=0,
    )

    res_list = wce_R(nknots=[1, 2, 3], **common_kwargs)
    assert res_list["nknots_grid"].tolist() == [1, 2, 3]
    assert res_list["best_nknots"][0] in (1, 2, 3)

    res_array = wce_R(nknots=np.array([1, 2, 3]), **common_kwargs)
    assert res_array["nknots_grid"].tolist() == [1, 2, 3]

    res_scalar = wce_R(nknots=2, **common_kwargs)
    assert res_scalar["nknots_grid"].tolist() == [2]
    assert res_scalar["best_nknots"][0] == 2


# HR() ----------------------------------------------------------------------
#
# Exposure vectors are in reverse chronological order (first value = current
# time). Shorter vectors are padded with zeros, longer ones truncated to
# `cutoff`. Where the reference WCE R package can be used (full-length
# vectors, no bootstrap), HR() is compared to WCE::HR.WCE(); otherwise to the
# HR computed by hand from the estimated weight function(s).

HR_CUTOFF = 90


@pytest.fixture(scope="module")
def hr_model():
    model = WCESurvivalAnalysis(cutoff=HR_CUTOFF, nknots=1)
    model.fit(
        dose=_dose, stop=_stop, start=_start, patient=_patient, event=_event
    )
    return model


@pytest.fixture(scope="module")
def hr_model_bootstrap():
    model = WCESurvivalAnalysis(
        cutoff=HR_CUTOFF, nknots=1, nbootstraps=20, batchsize=10
    )
    model.fit(
        dose=_dose, stop=_stop, start=_start, patient=_patient, event=_event
    )
    return model


@pytest.fixture(scope="module")
def hr_reference():
    """WCE::HR.WCE() on the same model fitted by the reference R package."""
    wce_ref = WCE.WCE(
        data=drugdata_r,
        analysis="Cox",
        nknots=1,
        cutoff=HR_CUTOFF,
        constrained=False,
        id="Id",
        event="Event",
        start="Start",
        stop="Stop",
        expos="dose",
    )
    hr_wce = ro.r("WCE::HR.WCE")

    def reference(vecnum, vecdenom):
        res = hr_wce(wce_ref, ro.FloatVector(vecnum), ro.FloatVector(vecdenom))
        return float(np.asarray(res).ravel()[0])

    return reference


def _weights(model):
    return model.risk_function_.reshape(-1).cpu().numpy()


def _hr_by_hand(weights, vecnum, vecdenom):
    return float(
        np.exp(np.dot(weights, np.asarray(vecnum) - np.asarray(vecdenom)))
    )


def _pad(x, n=HR_CUTOFF):
    return np.concatenate(
        [np.asarray(x, dtype=np.float64), np.zeros(n - len(x))]
    )


@pytest.mark.needs_keops()
def test_wce_hr_matches_r_wce_full_length(hr_model, hr_reference):
    rng = np.random.default_rng(42)
    scenarios = [
        (np.ones(HR_CUTOFF), np.zeros(HR_CUTOFF)),
        (_pad(np.full(10, 2.0)), np.zeros(HR_CUTOFF)),
        (np.r_[np.zeros(60), np.ones(30)], np.r_[np.ones(30), np.zeros(60)]),
        (rng.uniform(0, 3, HR_CUTOFF), rng.uniform(0, 3, HR_CUTOFF)),
    ]
    for vecnum, vecdenom in scenarios:
        hr = hr_model.HR(vecnum, vecdenom)
        assert set(hr) == {"HR"}
        assert hr["HR"] == pytest.approx(
            hr_reference(vecnum, vecdenom), rel=1e-4
        )
        assert hr["HR"] == pytest.approx(
            _hr_by_hand(_weights(hr_model), vecnum, vecdenom), rel=1e-12
        )


@pytest.mark.needs_keops()
def test_wce_hr_pads_shorter_vectors_with_zeros(hr_model, hr_reference):
    recent = np.ones(30)
    expected = hr_reference(_pad(recent), np.zeros(HR_CUTOFF))

    assert hr_model.HR(recent, np.zeros(HR_CUTOFF))["HR"] == pytest.approx(
        expected, rel=1e-4
    )
    # a scalar is a vector of length 1: padded too
    assert hr_model.HR(recent, 0)["HR"] == pytest.approx(expected, rel=1e-4)
    assert hr_model.HR(recent, 0)["HR"] == pytest.approx(
        _hr_by_hand(_weights(hr_model), _pad(recent), np.zeros(HR_CUTOFF)),
        rel=1e-12,
    )


@pytest.mark.needs_keops()
def test_wce_hr_truncates_longer_vectors(hr_model):
    expected = _hr_by_hand(
        _weights(hr_model), np.ones(HR_CUTOFF), np.zeros(HR_CUTOFF)
    )

    long_num = np.r_[np.ones(HR_CUTOFF), np.full(30, 5.0)]
    assert hr_model.HR(long_num, np.zeros(150))["HR"] == pytest.approx(
        expected, rel=1e-12
    )
    # exposures older than the cutoff have no effect
    older = np.r_[np.ones(HR_CUTOFF), np.full(30, 100.0)]
    assert hr_model.HR(older, 0)["HR"] == pytest.approx(expected, rel=1e-12)


@pytest.mark.needs_keops()
def test_wce_hr_mixed_lengths_lists_and_fractional_doses(
    hr_model, hr_reference
):
    vecnum = [1] * 30  # a plain list, shorter than the cutoff
    vecdenom = np.full(120, 0.5)  # fractional doses, longer than the cutoff
    full_num, full_den = _pad(vecnum), np.full(HR_CUTOFF, 0.5)

    hr = hr_model.HR(vecnum, vecdenom)["HR"]
    assert hr == pytest.approx(hr_reference(full_num, full_den), rel=1e-4)
    assert hr == pytest.approx(
        _hr_by_hand(_weights(hr_model), full_num, full_den), rel=1e-12
    )


@pytest.mark.needs_keops()
def test_wce_hr_bootstrap_interval_by_hand(hr_model_bootstrap):
    model = hr_model_bootstrap
    delta = _pad(np.ones(30))
    boot_weights = model.bootstrap_risk_functions_.squeeze(1).cpu().numpy()
    boot_hr = np.exp(boot_weights @ delta)

    for level in (0.95, 0.9):
        hr = model.HR(np.ones(30), 0, level=level)
        assert set(hr) == {"HR", "CI_lower", "CI_upper"}
        assert hr["HR"] == pytest.approx(
            _hr_by_hand(_weights(model), delta, np.zeros(HR_CUTOFF)), rel=1e-12
        )
        alpha = (1 - level) / 2
        assert hr["CI_lower"] == pytest.approx(
            np.quantile(boot_hr, alpha), rel=1e-12
        )
        assert hr["CI_upper"] == pytest.approx(
            np.quantile(boot_hr, 1 - alpha), rel=1e-12
        )
        assert hr["CI_lower"] < hr["CI_upper"]
