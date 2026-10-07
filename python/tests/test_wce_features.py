import pytest
import torch
from survivalgpu import float32, int32
from survivalgpu.utils import default_device
from survivalgpu.wce_features import bspline_atoms, wce_features_batch


@pytest.mark.needs_keops()
def test_bspline_atoms_shapes():
    """bspline_atoms returns (cutoff, F) features and (K,) knots.

    F = nknots + order + 1 (unconstrained), K = nknots + 2 + 2*order,
    per the docstrings of bspline_atoms / place_knots in wce_features.py.
    """
    order = 3
    nknots = 1
    cutoff = 20

    atoms, knots = bspline_atoms(
        cutoff=cutoff,
        nknots=nknots,
        order=order,
        dtype=float32,
        device=default_device,
    )

    F = nknots + order + 1
    K = nknots + 2 + 2 * order

    assert atoms.shape == (cutoff, F)
    assert knots.shape == (K,)
    assert torch.all(knots[1:] >= knots[:-1])
    assert knots[0].item() == -order
    assert knots[-1].item() == cutoff + order

    # B-spline atoms partition unity away from the boundary effects at the
    # very first/last samples:
    row_sums = atoms.sum(dim=1)
    assert torch.allclose(
        row_sums[order:-order],
        torch.ones_like(row_sums[order:-order]),
        atol=1e-4,
    )


@pytest.mark.needs_keops()
def test_wce_features_batch_shapes_and_values():
    """wce_features_batch reproduces bspline_atoms for a single dose at t=0."""
    order = 3
    nknots = 1
    cutoff = 20

    times = torch.arange(0, cutoff, device=default_device, dtype=int32)
    N = len(times)
    ids = torch.zeros(N, device=default_device, dtype=int32)
    doses = torch.zeros(N, device=default_device, dtype=float32)
    doses[times == 0] = 1

    features, knots = wce_features_batch(
        ids=ids,
        times=times,
        doses=doses,
        nknots=nknots,
        cutoff=cutoff,
        order=order,
        dtype=float32,
        device=default_device,
    )

    F = nknots + order + 1
    K = nknots + 2 + 2 * order

    assert features.shape == (N, F)
    assert knots.shape == (K,)

    # This construction is exactly what bspline_atoms does internally, so
    # results must match:
    atoms, atoms_knots = bspline_atoms(
        cutoff=cutoff,
        nknots=nknots,
        order=order,
        dtype=float32,
        device=default_device,
    )
    assert torch.allclose(features, atoms)
    assert torch.equal(knots, atoms_knots)


@pytest.mark.needs_keops()
def test_wce_features_batch_multi_patient_independence():
    """Two independent patients' doses must not leak into each other's features."""
    order = 3
    nknots = 1
    cutoff = 20

    times = torch.arange(-5, cutoff + 10, device=default_device, dtype=int32)
    N = len(times)
    times = torch.cat((times, times))
    ids = torch.cat(
        (
            torch.zeros(N, device=default_device, dtype=int32),
            torch.ones(N, device=default_device, dtype=int32),
        )
    )
    doses = torch.zeros(2 * N, device=default_device, dtype=float32)
    doses[(times == 0) & (ids == 0)] = 1
    doses[(times == 5) & (ids == 1)] = 1
    doses[(times == 10) & (ids == 1)] = 2

    features, knots = wce_features_batch(
        ids=ids,
        times=times,
        doses=doses,
        nknots=nknots,
        cutoff=cutoff,
        order=order,
        dtype=float32,
        device=default_device,
    )

    F = nknots + order + 1
    assert features.shape == (2 * N, F)
    assert knots.shape == (nknots + 2 + 2 * order,)

    # Patient 0's only dose is at t=0: the B-Spline window rule requires
    # 1 <= (t + 1) - dose_time < cutoff + 1, so times < 0 must have zero
    # features regardless of patient 1's (later) doses:
    before_dose_0 = (ids == 0) & (times < 0)
    assert torch.allclose(
        features[before_dose_0], torch.zeros_like(features[before_dose_0])
    )
