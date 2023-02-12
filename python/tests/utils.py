import torch
import numpy as np

# Functions that we are trying to extend:
from survivalgpu import coxph_torch
from survivalgpu.wce import constrain
from survivalgpu.wce_features import wce_features_batch, bspline_atoms


def numpy(x):
    return x.cpu().numpy()


def form(x):
    return np.array_str(numpy(x), precision=3)


def wce_features(*, doses, times, nknots, cutoff, order, constrained=None):
    """Compute the WCE features for a batch of patients.

    The total number of WCE features is equal to:
        nknots + order + 1 if constrained is None,
        nknots + 2         if constrained is "Left" or "Right".

    Args:
        doses (Tensor): (Drugs, Patients, Times) float32 Tensor.
        times (Tensor): (Times,) int32 Tensor.
        nknots (int): number of knots for the B-splines.
        cutoff (int): maximum time for the B-splines.
        order (int): order of the B-splines.
        constrained (str, optional): whether the B-splines should be constrained.
            Defaults to None (i.e. no constrain).
            Other options are "Left" (the drug has no immediate effect on the risk)
            and "Right" (the drug has no effect on the risk around the cutoff time).

    Returns:
        dict: contains Tensors that describe the WCE features for a batch of patients:
            - "exposures": a (Drugs, Patients, Times, Features) float32 Tensor.
            - "knots": a (nknots + !!!,) int32 Tensor with the knot positions.
            - "atoms": a (cutoff, Features) float32 Tensor that contains the sampled
                values of the fundamental B-splines functions on [0, 1, ..., cutoff-1].
            - "areas": a (Features,) float32 Tensor with the areas of each B-spline atom.
            - "cutoff": int that keeps track of the maximum time.
    """

    Drugs, Patients, Times = doses.shape
    device = doses.device
    N = Drugs * Patients * Times

    # The drugs and patients are simple integer labels:
    drugs = torch.arange(Drugs, dtype=torch.int32, device=device)
    patients = torch.arange(Patients, dtype=torch.int32, device=device)

    # (Drugs, Patients)
    ids = Patients * drugs.view(Drugs, 1) + patients.view(1, Patients)
    ids = ids.view(Drugs, Patients, 1).tile([1, 1, Times])

    exposures, knots = wce_features_batch(
        ids=ids.view(N),
        times=times.tile([Drugs * Patients]).view(N),
        doses=doses.view(N),
        nknots=nknots,
        cutoff=cutoff,
        order=order,
    )
    assert exposures.shape == (Drugs * Patients * Times, nknots + order + 1)

    # Remove some of the covariates if required:
    exposures = constrain(features=exposures, constrained=constrained, order=order)

    if constrained is None:
        Features = nknots + order + 1
    else:
        Features = nknots + 2

    exposures = exposures.view(Drugs, Patients, Times, Features)
    assert exposures.shape == (Drugs, Patients, Times, Features)

    # Compute the b-spline atom function on the interval [0, cutoff]:
    atoms, _ = bspline_atoms(cutoff=cutoff, order=order, knots=knots)
    atoms = constrain(features=atoms, constrained=constrained, order=order)
    assert atoms.shape == (cutoff, Features)

    # Compute the "risk area under the curve" that is associated to those atoms:
    areas = atoms.sum(0)  # (Features)
    assert areas.shape == (Features,)

    return {
        "exposures": exposures,
        "knots": knots,
        "atoms": atoms,
        "areas": areas,
        "cutoff": cutoff,
    }


def coxph_fit(
    *,
    exposures,
    times,
    events,
    areas,
    permutation=None,
    interest_drug=None,
    bootstrap=1,
):
    """Fits a Cox model to the exposure covariates.

    Args:
        exposures (Tensor): (Drugs, Patients, Times, Features) float32 Tensor
            of time-dependent covariates.
        times (Tensor): (Patients, Times) int32 Tensor.
        events (Tensor): (Patients, Times) int32 Tensor whose values are equal to
            0 if everything is all right, i.e. the patient is still "alive",
            1 if the patient is "dying" at this exact moment,
            2+ if this consumption happened after the event of interest,
                and should therefore be removed from the survival analysis.
        areas (Tensor): (Features,) float32 Tensor that contains the areas of the B-spline atoms.
        permutation (Tensor, optional): (Patients,) int32 Tensor that may contain a random
            permutation of the patient indices, to apply in a permutation test
            to destroy the correlation between the covariates and the events.
            Defaults to None (i.e. no permutation).
        interest_drug (int, optional): index of the drug of interest.
            Defaults to None (i.e. all drugs are considered).
        bootstrap (int, optional): number of bootstrap samples.
            Defaults to 1 (i.e. no bootstrap).

    Returns:
        dict: contains the fitted model parameters, as described in the survivalGPU package.
            As an additional attribute, we also include "risk", which is a
            (bootstrap, Drugs) Tensor that contains the estimated total risk associated
            to each drug - i.e. the total area under the risk function.
            These are the logarithms of the "hazard ratios" (HR).
    """
    Drugs, Patients, Times, Features = exposures.shape
    assert areas.shape == (Features,)

    # We may apply a random permutation to the second axis (=patients) of the array
    # of exposures. This is to make sure that the events are not correlated
    # with the covariates.
    if permutation is not None:
        exposures = exposures[:, permutation, :, :]

    # Note that the Cox model does not care about the patient indices
    # (just who lives and who dies at each time point), so we can flatten
    # the array of exposures.
    N = Patients * Times
    exposures = exposures.view(Drugs, N, Features)
    events = events.view(N)

    # We remove all cell with "event >= 2", i.e. after death: -------------------------------
    # N.B.: this is very important, and skipping this step would be an easy mistake to make!
    mask = events <= 1
    N = mask.sum()

    events = events[mask]
    times = times.tile([Patients])[mask]
    exposures = exposures[:, mask, :]

    assert exposures.shape == (Drugs, N, Features)
    assert times.shape == (N,)
    assert events.shape == (N,)

    # We may only consider the drug of interest, e.g. for in-depth visualization with
    # bootstraps on the drugs that have been identified as "dangerous": --------------------
    if interest_drug is not None:
        exposures = exposures[interest_drug, :, :]

    # Fit the CoxPH model on all Drugs in parallel: ------------------------------------------
    coxph_output = coxph_torch(
        x=exposures,
        times=times,
        deaths=events,
        bootstrap=bootstrap,
    )

    # (bootstrap, Drugs) = (bootstrap, Drugs, Features) @ (Features)
    coxph_output["risk"] = coxph_output["coef"] @ areas

    assert coxph_output["coef"].shape == (bootstrap, Drugs, Features)
    assert coxph_output["risk"].shape == (bootstrap, Drugs)

    return coxph_output
