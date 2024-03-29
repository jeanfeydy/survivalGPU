import pytest
import numpy as np
from numpy.testing import assert_allclose
import torch

from survivalgpu import coxph_R
from survivalgpu.utils import numpy
from survivalgpu.optimizers import newton


np.set_printoptions(precision=4)

# 1. Sanity check ======================================
data_csv = np.array(
    [
        # Time, Death, Covars
        [1, 0, -1.0, 0.0],
        [1, 0, 4.0, 4.0],
        [1, 1, 0.0, 2.0],
        [2, 0, 4.0, 2.0],
        [2, 0, 0.0, 0.0],
        [3, 1, 4.0, 1.0],
    ]
)
data = {
    "stop": data_csv[:, 0],
    "death": data_csv[:, 1],
    "covar1": data_csv[:, 2],
    "covar2": data_csv[:, 3],
}

for ties in ["efron", "breslow"]:
    for doscale in [True, False]:
        print(f"\nties = {ties}, doscale = {doscale} ========")
        res = coxph_R(
            data,
            "stop",
            "death",
            ["covar1", "covar2"],
            bootstrap=1,
            ties=ties,
            doscale=doscale,
            profile=None,
        )
        for key, item in res.items():
            print(f"{key}:")
            print(item)



def test_newton_convex():
    """Check that Newton's method works on a separable convex problem."""

    def loss(b):
        loss_1 = (b[:, 0] - 1) ** 2 + b[:, 0].exp()
        loss_2 = (b[:, 1] - 3) ** 4 + (-2 * b[:, 1]).exp()
        loss_3 = 1 + b[:, 2] ** 2 + b[:, 2].exp()

        return loss_1 + loss_2 + loss_3

    res = newton(
        loss=loss,
        start=torch.zeros(1, 3),
        maxiter=50,
        verbosity=0,
    )

    assert_allclose(numpy(res["x"][0]), np.array([0.31492, 3.1005, -0.35173]), 1e-2)

    print(f"Three simple problems:")
    print(f"Newton solutions: {numpy(res['x'][0])}")
    print(f"Should be equal to 0.31492  3.1005  -0.35173")
    print("")


def test_newton_coxph():
    """Checks that Newton's method works on a mini-CoxPH problem."""

    def loss(b):
        return (b * torch.FloatTensor([[0, -1, 4]])).logsumexp(dim=1)

    res = newton(
        loss=loss,
        start=torch.zeros(1, 1),
        maxiter=50,
        verbosity=0,
    )

    assert numpy(res["x"][0]) == pytest.approx(-0.275057, 1e-2)
    assert numpy(res["fun"]) == pytest.approx(0.97509, 1e-2)

    print(f"Mini-CoxPH problem:")
    print(f"Newton solution: {numpy(res['x'][0])}")
    print(f"Value: {numpy(res['fun'])}")
    print(f"Should be equal to -0.275057 and 0.97509")
    print("")


if __name__ == "__main__":
    test_newton_convex()
    test_newton_coxph()