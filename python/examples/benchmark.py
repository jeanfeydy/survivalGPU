import numpy as np
from survivalgpu import coxph_numpy, use_cuda
from survivalgpu.utils import numpy, timer

np.set_printoptions(precision=4)


def benchmark(data, bootstrap=1, alpha=0.0):
    backends = ["torch"]

    for backend in backends:
        print(f"Backend: {backend} **************")

        for use_gpu in [False] + ([True] if use_cuda else []):
            data_x = data[:, 2:].astype(np.float64)
            data_stop = data[:, 0].astype(np.int64)
            data_deaths = data[:, 1].astype(np.int64)

            device = "cuda" if use_gpu else "cpu"

            start = timer()

            out = coxph_numpy(
                x=data_x,
                start=None,
                stop=data_stop,
                deaths=data_deaths,
                ties="breslow",
                bootstrap=bootstrap,
                maxiter=20,
                verbosity=0,
                alpha=alpha,
                device=device,
            )

            end = timer()

            print(
                f"GPU={use_gpu!s:5}, parameter: {numpy(out['coef'])[0]}, "
                f"value: {numpy(out['loglik'])[0]}, "
                f"{end - start:.3f}s"
            )


print(f"Are we using a GPU? {use_cuda}")

# 1. Sanity check ======================================
data_1 = np.array(
    [
        # Time, Death, Covars
        [1, 0, -1.0],
        [1, 0, 4.0],
        [1, 1, 0.0],
        [2, 0, -3.0],
        [2, 0, 3.0],
    ]
)

print("Minimal test with several backends ================")
print("The output should be equal to -0.27726 and 0.97433.")
benchmark(data_1)
print("")

# 2. Mini benchmark ====================================
n_points = 100000 if use_cuda else 100
n_times = 10000 if use_cuda else 50
n_features = 5
death_ratio = 0.1
n_bootstraps = 100

rng = np.random.default_rng()

data_times = rng.integers(n_times, size=(n_points, 1)) * 1.0
data_deaths = (rng.uniform(size=(n_points, 1)) < death_ratio) * 1.0
data_covars = rng.standard_normal(size=(n_points, n_features))

data_2 = np.concatenate((data_times, data_deaths, data_covars), axis=1)
print(
    f"Mini-benchmark with {n_bootstraps} bootstraps, {n_points} points, {n_features} features,"
)
print(f"{n_times} death times and a death ratio of {death_ratio:.2f}")

print("")
