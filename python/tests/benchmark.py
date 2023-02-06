import numpy as np
import torch

from survivalgpu import coxph_torch, use_cuda, float32, int32
from survivalgpu.utils import timer, numpy


np.set_printoptions(precision=4)


def benchmark(data, bootstrap=1, backends=None, alpha=0.0):
    if backends is None:
        backends = ([] if use_cuda else ["torch"]) + ["pyg", "coo", "csr"]

    for backend in backends:
        print(f"Backend: {backend} **************")

        for use_gpu in [False] + ([True] if use_cuda else []):
            data_x = torch.tensor(data[:, 2:], dtype=float32)
            data_times = torch.tensor(data[:, 0], dtype=int32)
            data_deaths = torch.tensor(data[:, 1], dtype=int32)

            if use_gpu:
                data_x = data_x.cuda()
                data_times = data_times.cuda()
                data_deaths = data_deaths.cuda()

            start = timer()

            out = coxph_torch(
                x=data_x,
                times=data_times,
                deaths=data_deaths,
                ties="breslow",
                backend=backend,
                bootstrap=bootstrap,
                maxiter=20,
                verbosity=0,
                alpha=alpha,
            )

            end = timer()

            print(
                f"GPU={str(use_gpu):5}, parameter: {numpy(out['coef'])[0]}, "
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

data_times = np.random.randint(n_times, size=(n_points, 1)) * 1.0
data_deaths = (np.random.uniform(size=(n_points, 1)) < death_ratio) * 1.0
data_covars = np.random.randn(n_points, n_features)

data_2 = np.concatenate((data_times, data_deaths, data_covars), axis=1)
print(
    f"Mini-benchmark with {n_bootstraps} bootstraps, {n_points} points, {n_features} features,"
)
print(f"{n_times} death times and a death ratio of {death_ratio:.2f}")

benchmark(data_2, backends=["pyg", "coo", "csr"], bootstrap=n_bootstraps, alpha=0.1)
print("")
