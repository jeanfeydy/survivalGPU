"""Shared synthetic datasets used by the examples in this folder.

Both helpers below use survivalGPU's own simulator (`simulate_dataset`) so
that the "true" coefficients / hazard ratios used to generate the data are
known in advance, which makes it easy to sanity-check what each example
prints.
"""

import numpy as np
from survivalgpu import ConstantCovariate, WCECovariate, simulate_dataset

# Coefficients used to simulate the CoxPH dataset. `simulate_dataset` picks
# covariate trajectories so that the fitted model should recover values
# close to these.
COXPH_TRUE_COEF = {"sex": 0.7, "biomarker": 0.4}

# Target hazard ratio used to simulate the WCE dataset (dose == cutoff vs.
# dose == 0 over the whole exposure window).
WCE_TRUE_HR = 3.0


def coxph_dataset(n_patients: int = 300, max_time: int = 200, seed: int = 0):
    """Simulates a simple (start, stop] CoxPH dataset with two covariates.

    Returns:
        dict of float64/int64 NumPy arrays with keys "covariates", "start",
        "stop", "event", "patient" — ready to be passed to
        `CoxPHSurvivalAnalysis.fit()` or `coxph_numpy()`.
    """
    sex = ConstantCovariate(
        name="sex", values=[0, 1], weights=[1, 1], coef=COXPH_TRUE_COEF["sex"]
    )
    biomarker = ConstantCovariate(
        name="biomarker",
        values=[0, 1, 2],
        weights=[1, 1, 1],
        coef=COXPH_TRUE_COEF["biomarker"],
    )

    df = simulate_dataset(
        max_time=max_time,
        n_patients=n_patients,
        list_covariates=[sex, biomarker],
        compress=True,
        seed=seed,
    )

    return {
        "covariates": df[["sex", "biomarker"]].to_numpy(dtype=np.float64),
        "start": df["start"].to_numpy(dtype=np.int64),
        "stop": df["stop"].to_numpy(dtype=np.int64),
        "event": df["events"].to_numpy(dtype=np.int64),
        "patient": df["patients"].to_numpy(dtype=np.int64),
    }


def wce_dataset(
    n_patients: int = 300,
    max_time: int = 200,
    seed: int = 1,
):
    """Simulates a drug-exposure dataset for the WCE model.

    A single time-varying "dose" covariate follows a "bi_linear_scenario"
    risk shape (the risk of the drug decreases linearly with time-since-dose
    over the `cutoff` window) with a known target hazard ratio
    `WCE_TRUE_HR`. A constant "sex" covariate is added on top, to also
    illustrate mixing WCE and standard CoxPH covariates in a single model.

    Note: unlike `coxph_dataset`, this dataset is *not* compressed, since
    WCESurvivalAnalysis currently requires unit-length (start, start+1]
    intervals.

    Returns:
        dict of float64/int64 NumPy arrays with keys "dose", "covariates"
        (the "sex" column), "start", "stop", "event", "patient".
    """
    dose = WCECovariate(
        name="dose",
        values=[1, 1.5, 2, 2.5, 3],
        scenario_name="bi_linear_scenario",
        HR_target=WCE_TRUE_HR,
    )
    sex = ConstantCovariate(name="sex", values=[0, 1], weights=[1, 1], coef=0.5)

    df = simulate_dataset(
        max_time=max_time,
        n_patients=n_patients,
        list_covariates=[dose, sex],
        compress=False,
        seed=seed,
    )

    return {
        "dose": df["dose"].to_numpy(dtype=np.float64),
        "covariates": df[["sex"]].to_numpy(dtype=np.float64),
        "start": df["start"].to_numpy(dtype=np.int64),
        "stop": df["stop"].to_numpy(dtype=np.int64),
        "event": df["events"].to_numpy(dtype=np.int64),
        "patient": df["patients"].to_numpy(dtype=np.int64),
    }
