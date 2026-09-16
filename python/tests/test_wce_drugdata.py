import numpy as np
import pytest
import rpy2.robjects as ro
import rpy2.robjects.packages as rpackages
import torch
from rpy2.robjects import conversion, pandas2ri
from survivalgpu import WCESurvivalAnalysis

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
