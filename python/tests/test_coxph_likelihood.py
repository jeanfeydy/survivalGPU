import numpy as np
import torch
from hypothesis import given
from hypothesis import strategies as st
from survivalgpu.autodiff import derivatives_012
from survivalgpu.bootstrap import Resampling
from survivalgpu.coxph_likelihood import coxph_objective
from survivalgpu.torch_datasets import TorchSurvivalDataset

np.set_printoptions(precision=4)

SUPPORTED_TIES = ["breslow", "efron"]

if torch.cuda.is_available():
    st_device = st.sampled_from(["cpu", "cuda"])
else:
    st_device = st.just("cpu")


# Small datasets, ordered from simplest to most complex
# fmt: off
examples = [
    dict(
        intervals= [
            # Patient, Start, Stop, Event, Covar
            [       0,     0,    1,     1,    1.],
        ],
        patients=[
            # Batch, Strata
            [     0,      0],
        ],
        bootstraps=[
            [0],
        ],
        beta=[
            [[1.]],
        ],
        breslow=dict(
            loss = [
                [0.],
            ],
            grad = [
                [0.],
            ],
            hessian = [
                [[0.]],
            ],
        ),
        efron=dict(
            loss = [
                [0.],
            ],
            grad = [
                [0.],
            ],
            hessian = [
                [[0.]],
            ],
        ),
    ),
]
# fmt: on


@given(
    data=st.sampled_from(examples),
    ties=st.sampled_from(SUPPORTED_TIES),
    device=st_device,
)
def test_loss_grad_hessian(*, data, ties, device):
    intervals = torch.tensor(
        data["intervals"], dtype=torch.int64, device=device
    )[:, :4]
    patients = torch.tensor(data["patients"], dtype=torch.int64, device=device)
    bootstraps = torch.tensor(
        data["bootstraps"], dtype=torch.int64, device=device
    )
    covars = torch.tensor(
        data["intervals"], dtype=torch.float32, device=device
    )[:, 4:]
    beta = torch.tensor(
        data["beta"], dtype=torch.float32, device=device, requires_grad=True
    )

    gt_loss = torch.tensor(
        data[ties]["loss"], dtype=torch.float32, device=device
    )
    gt_grad = torch.tensor(
        data[ties]["grad"], dtype=torch.float32, device=device
    )
    gt_hessian = torch.tensor(
        data[ties]["hessian"], dtype=torch.float32, device=device
    )

    dataset = TorchSurvivalDataset(
        batch=patients[:, 0],
        strata=patients[:, 1],
        patient=intervals[:, 0],
        start=intervals[:, 1],
        stop=intervals[:, 2],
        event=intervals[:, 3],
        covariates=covars,
    ).sort()

    bootstrap = Resampling(
        indices=bootstraps,
        patient=patients[:, 0],
    )
    B = len(bootstrap)
    n_batch = dataset.n_batch
    D = dataset.n_covariates

    def f(coef):
        return coxph_objective(
            coef=coef.view(B, n_batch, D),
            scales=None,
            dataset=dataset,
            ties=ties,
            bootstrap=bootstrap,
            l2_reg=0.0,
        ).view(B * n_batch)

    f_grad_hessian = derivatives_012(f)

    with torch.autograd.detect_anomaly():
        loss, grad, hessian = f_grad_hessian(beta.view(B * n_batch, D))

    assert torch.allclose(loss, gt_loss, rtol=1e-4, atol=1e-4)
    assert torch.allclose(grad, gt_grad, rtol=1e-4, atol=1e-4)
    assert torch.allclose(hessian, gt_hessian, rtol=1e-4, atol=1e-4)
