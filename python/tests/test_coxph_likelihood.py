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
    # First example: just one patient, who dies at time 1.
    # Since there is no comparison between death and survival, the loss is uniformly 0.
    dict(
        intervals= [
            # Patient, Start, Stop, Event, Covar
            [       0,     0,    1,     1,    1.],
        ],
        patients=[
            # Batch, Strata
            [     0,      0],
        ],
        # Just one bootstrap, with patient 0 picked once.
        bootstraps=[[0]],
        # Evaluate at beta = (1,)
        beta=[ [[1.]] ],
        breslow=dict(
            loss = [[0.]],
            grad = [[0.]],
            hessian = [[[0.]]],
        ),
        efron=dict(
            loss = [[0.]],
            grad = [[0.]],
            hessian = [[[0.]]],
        ),
    ),

    # Second example: two patients, one dies and one survives at time 1.
    dict(
        intervals= [
            # Patient, Start, Stop, Event, Covar
            [       1,     0,    1,     1,    2.],  # Spice things up with patient 1 on row 0
            [       0,     0,    1,     0,   -1.],  # and patient 0 on row 1
        ],
        patients=[
            # Batch, Strata
            [     0,      0],  # Patient 0
            [     0,      0],  # Patient 1
        ],
        # One balanced bootstrap (2 + 2), and one unbalanced (1 + 3)
        bootstraps=[
            [0, 0, 1, 1],
            [0, 1, 1, 1],
        ],
        # Evaluate at b = beta = (1,) in the first bootstrap, beta = (2,) in the second bootstrap
        beta=[ [[1.]], [[2.]] ],

        # With these intervals, the Breslow loss for the first bootstrap is:
        # 2 * log( 2 * exp(-b) + 2 * exp(2b) ) - 2 * 2b
        # with 1st derivative:
        # - 6 / (exp(3b) + 1)
        # and 2nd derivative:
        # 18 * exp(3b) / (exp(3b) + 1)^2
        # to be evaluated at b=1
        #
        # For the second bootstrap, the loss is:
        # 3 * log( 1 * exp(-b) + 3 * exp(2b) ) - 3 * 2b
        # with 1st derivative:
        # -9 / (3 * exp(3b) + 1)
        # and 2nd derivative:
        # 81 * exp(3b) / (3 * exp(3b) + 1)^2
        # to be evaluated at b=2
        breslow=dict(
            loss = [1.4835, 3.2983],
            grad = [[-0.2846], [-0.007430]],
            hessian = [[[0.8132]], [[0.022272]]],
        ),
        # With these intervals, the Efron loss for the first bootstrap is:
        #   log( 2 * exp(-b) + 2 * exp(2b) )
        # + log( 2 * exp(-b) + 1 * exp(2b) ) - 2 * 2b
        # to be evaluated at b=1
        #
        # For the second bootstrap, the loss is:
        #   log( 1 * exp(-b) + 3 * exp(2b) )
        # + log( 1 * exp(-b) + 2 * exp(2b) )
        # + log( 1 * exp(-b) + 1 * exp(2b) ) - 3 * 2b
        # to be evaluated at b=2
        efron=dict(
            loss = [0.836657, 1.7963],
            grad = [[-0.413949], [-0.013608]],
            hessian = [[[1.147798]], [[0.04074935]]],
        ),
    )
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
    patient_data = torch.tensor(
        data["patients"], dtype=torch.int64, device=device
    )
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
        batch=patient_data[:, 0],
        strata=patient_data[:, 1],
        patient=intervals[:, 0],
        start=intervals[:, 1],
        stop=intervals[:, 2],
        event=intervals[:, 3],
        covariates=covars,
    ).sort()

    bootstrap = Resampling(
        indices=bootstraps,
        patient=dataset.patient,
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

    # with torch.autograd.detect_anomaly():
    loss, grad, hessian = f_grad_hessian(beta.view(B * n_batch, D))

    assert torch.allclose(loss, gt_loss, rtol=1e-3, atol=1e-4)
    assert torch.allclose(grad, gt_grad, rtol=1e-3, atol=1e-4)
    assert torch.allclose(hessian, gt_hessian, rtol=1e-3, atol=1e-4)
