import torch
import time
from survivalgpu import CoxPHSurvivalAnalysis
from survivalgpu.datasets import simple_dataset


def benchmark_coxph_simple(
    *, n_covariates, n_patients, n_batch=1, n_strata=1, max_duration=10
):
    ds = simple_dataset(
        n_covariates=n_covariates,
        n_patients=n_patients,
        n_batch=n_batch,
        n_strata=n_strata,
        max_duration=max_duration,
        ensure_one_life=True,
        ensure_one_death=True,
        unit_length_intervals=True,
    )

    model = CoxPHSurvivalAnalysis(ties="breslow", alpha=0.01)

    time_start = time.time()
    model.fit(
        covariates=ds.covariates,
        start=ds.start,
        stop=ds.stop,
        event=ds.event,
        batch=ds.batch,
        strata=ds.strata,
    )
    time_end = time.time()
    print(
        f"{n_covariates} covariates, {n_patients:8,} patients -- time = {time_end - time_start:.2e}s"
    )


for n_patients in [1000, 10000, 100000]:
    benchmark_coxph_simple(
        n_covariates=5,
        n_patients=n_patients,
    )
