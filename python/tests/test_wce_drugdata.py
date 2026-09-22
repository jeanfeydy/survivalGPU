import numpy as np
import pytest
import rpy2.robjects as ro
import rpy2.robjects.packages as rpackages
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
