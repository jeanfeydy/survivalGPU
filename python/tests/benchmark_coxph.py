import torch
import time
from survivalgpu import CoxPHSurvivalAnalysis
from survivalgpu.datasets import simple_dataset
import functools

torch.use_deterministic_algorithms(False)

import numpy as np

# Set numpy print options to 4 digits
np.set_printoptions(precision=4)


@functools.cache
def my_dataset(
    *,
    n_covariates,
    n_patients,
    n_batch,
    n_strata,
    max_duration,
):
    return simple_dataset(
        n_covariates=n_covariates,
        n_patients=n_patients,
        n_batch=n_batch,
        n_strata=n_strata,
        max_duration=max_duration,
        ensure_one_life=True,
        ensure_one_death=True,
        unit_length_intervals=True,
    )


def benchmark_coxph_simple(
    *,
    n_covariates,
    n_patients,
    n_batch=1,
    n_strata=1,
    max_duration=None,
    backend="csr",
    maxiter=20,
):
    if max_duration is None:
        max_duration = n_patients

    ds = my_dataset(
        n_covariates=n_covariates,
        n_patients=n_patients,
        n_batch=n_batch,
        n_strata=n_strata,
        max_duration=max_duration,
    )

    model = CoxPHSurvivalAnalysis(
        ties="breslow", alpha=0.01, backend=backend, maxiter=maxiter
    )

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
        f"{n_covariates} covariates, {n_patients:8,} patients -- time = {time_end - time_start:3.3f}s"
    )
    print(model.coef_)


for backend in ["torch", "pyg", "coo", "csr"]:
    print("backend:", backend)
    for n_patients in [10000]:  # [1000, 10000, 100000]:
        benchmark_coxph_simple(
            n_covariates=5,
            n_patients=n_patients,
            backend=backend,
            maxiter=3,
        )

    if False:
        from torch.profiler import profile, record_function, ProfilerActivity

        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

        myprof = profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
        )

        with myprof as prof:
            N, D = n_patients, 5
            benchmark_coxph_simple(
                n_covariates=D,
                n_patients=N,
                backend=backend,
                maxiter=3,
            )

        # Create an "output/" foler if it doesn't exist
        import os

        if not os.path.exists("output"):
            os.makedirs("output")

        # Export to chrome://tracing
        prof.export_chrome_trace(f"output/trace_{backend}_{N}_{D}.json")
        prof.export_stacks(
            f"output/stacks_{backend}_{N}_{D}.txt", "self_cuda_time_total"
        )
