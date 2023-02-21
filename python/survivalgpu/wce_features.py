"""This file implements the core numerical routines of the WCE package.

We rely on KeOps to compute convolutions with a collection of B-Spline kernels,
at arbitrary time sampling locations.

TODO:
  * Implement a fallback mode that relies on a pure PyTorch implementation 
    when KeOps is not available.
"""


import numpy as np
import torch

from pykeops.torch import LazyTensor

from .utils import device, float32, int32


def place_knots(*, cutoff, nknots, order):
    """Returns a list of (nknots + 2 + 2*order) knot positions for the B-Splines model.

    The number of knots accounts for the fact that, following the conventions of the
    WCE R package, we place:
    - `nknots` inside the time window [0, cutoff] at regular intervals.
    - 1+1 knots at both ends of the time window (0 and cutoff).
    - order+order knots for "padding" at both ends of the time window.

    For instance, if order = 3, cutoff = 90 and nknots = 1,
    knots = [-3 -2 -1  0 46 90 91 92 93]  (length = 1 + 2 + 6 = 9)

    Please note that returning e.g. [0, 0, 0, 0, 46, 90, 90, 90, 90, 90] would be
    cleaner from a mathematical perspective - but for the sake of compatibility
    with the WCE R package, we stick to this "counting" convention.

    Args:
        cutoff (int): length of the observation window.
        nknots (int): number of inner knots.

    Returns:
        ((K,) array): (nknots + 2*order + 2) values for the knot positions.
    """
    # Place nknots+1 equispaced values in [1, 2, ..., cutoff], at integer positions.
    knots = np.round(
        np.quantile(1 + np.arange(cutoff), np.arange(nknots + 1) / (nknots + 1)), 0
    )
    # And remove the first quantile:
    knots = knots[1:]  # (nknots,)
    # The code above ensures that if nknots=1, our knot will fall around cutoff/2.

    # Pad the knot vectors with:
    # - [-order, -(order-1), ..., 0] to the left
    # - [cutoff, cutoff+1, ..., cutoff+order] to the right
    ends = np.arange(order + 1)
    knots = np.concatenate(
        (-ends[::-1], knots, cutoff + ends)
    )  # (nknots + 2 + 2*order,)
    return knots


# KeOps computation of the B-Spline covariates ===========================================


def ranges_slices(batch):
    """Helper function for the diagonal ranges function."""
    Ns = batch.bincount()
    indices = Ns.cumsum(0)
    ranges = torch.cat((0 * indices[:1], indices))
    ranges = (
        torch.stack((ranges[:-1], ranges[1:])).t().int().contiguous().to(batch.device)
    )
    slices = (1 + torch.arange(len(Ns))).int().to(batch.device)

    return ranges, slices


def diagonal_ranges(batch_x=None, batch_y=None):
    """Encodes the block-diagonal structure associated to a pair of batch vectors."""

    if batch_x is None and batch_y is None:
        return None  # No batch processing
    elif batch_y is None:
        batch_y = batch_x  # "symmetric" case

    ranges_x, slices_x = ranges_slices(batch_x)
    ranges_y, slices_y = ranges_slices(batch_y)

    return ranges_x, slices_x, ranges_y, ranges_y, slices_y, ranges_x


def bspline_conv(
    *,
    target_times,
    target_ids,
    source_times,
    source_weights,
    source_ids,
    knots,
    window,
    order=3,
):
    """Performs a convolution (= weighted sum) with a collection of B-Spline kernels.

    If t[i] is a sampling time and if the drug consumption history of a patient
    can be described using a finite collection of dose times s[j] and dose weights w[j],
    then for any index k in [1, K - order - 1], we compute:

    out[i, k] = Sum_{j} (
          B_k(t[i] - s[j])
        * 1_{window[0] <= t[i] - s[j] < window[1]}
        * w[j]
    )

    where B_k(x) denotes the k-th B-spline function of order "order"
    associated to our knots evaluted at x.
    These correspond to piecewise constant, linear, quadratic and cubic functions
    for order = 0, 1, 2 and 3, respectively.
    We evaluate the B-Spline functions using the recursive De Boor algorithm.

    This function expects the target times t[i], source times s[j] and source weights w[j]
    to be "stacked" in three distinct vectors. We use vectors of non-decreasing
    integer values to "group" these times by patient id.

    This follows the convention of the PyTorch_Geometric library as detailed here:
    https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html#mini-batches

    Args:
        target_times ((N,) tensor): the times at which we want to evaluate
            the B-Spline features.
        target_ids ((N,) int tensor): "batch" vector of non-decreasing,
            contiguous values from 0 to N_subjects - 1.
            These correspond to patient ids for the "target" times t[i].

        source_times ((M,) tensor): the times that correspond to the input
            weights.
        source_weights ((M,) tensor): weights for the translated B-Spline
            kernels. In practice, these weights correspond to drug doses.
        source_ids ((M,) int tensor): "batch" vector of increasing,
            contiguous values from 0 to N_subjects - 1.
            These correspond to patient ids for the "source" times s[j]
            and weights w[j].

        knots ((K,) tensor): positions for the B-Spline knots.
        window ((2,) tensor): start and end times for the observation window.
            Typically, this is equal to [1, cutoff+1].
        order (int, optional): order of the B-Spline piece-wise polynomials.
            Should be >= 0. Defaults to 3 (= cubic splines).

    Returns:
        features ((N, K - order - 1) tensor): the values of the weighted sums
            of B-Spline functions at the N target times.
    """
    # N.B.: In the original WCE package, the BSpline basis is created
    #       on a domain x = [1, ..., cutoff] instead of [0, ..., cutoff].
    #       As a consequence, we should offset the event times
    #       by 1 to retrieve the exact same results.
    events_i = LazyTensor((target_times + 1).float().view(-1, 1, 1))  # (N,1,1)
    doses_times_j = LazyTensor(source_times.float().view(1, -1, 1))  # (1,M,1)
    doses_values_j = LazyTensor(source_weights.float().view(1, -1, 1))  # (1,M,1)

    # The constant parameters are simply encoded as vectors:
    knots_ = LazyTensor(knots.float().view(1, 1, -1))  # (1,1,K)

    # Our rule for the "cutoff" window will be to ensure that
    # 1 <= my_ev_i - stop_j < cutoff + 1
    cut_ = LazyTensor(window.float().view(1, 1, -1))  # (1,1,2)

    # Symbolic KeOps computation.
    # We encode the cutoff window as a B-Spline of order 0:
    window_ij = cut_.bspline(events_i - doses_times_j, 0)  # (N,M,1)
    # We use the general B-Spline KeOps formula to compute the features in parallel:
    atoms_ij = knots_.bspline(events_i - doses_times_j, order)  # (N,M,K-order-1)
    full_ij = window_ij * atoms_ij * doses_values_j  # (N,M,K-order-1)

    # Block-diagonal ranges:
    full_ij.ranges = diagonal_ranges(target_ids, source_ids)

    # Sum over the source index "j":
    return full_ij.sum(1)  # (N,K-order-1)


def wce_features_batch(*, ids, times, doses, nknots, cutoff, order=3, knots=None):
    """This function is equivalent to a parallel application of the .wcecalc method from the WCE package.

    The number of B-spline covariates is equal to
     F = (K - order - 1)
    where K is the length of `knots` if it is not None,
    or is equal to (nknots + 2 + 2*order) otherwise (leading to F = nknots+order+1).

    From R, calling:

    kal <- data.frame(wce_features_batch(ev, data, length(Id), cutoff, kev[[j]]))


    Is equivalent to the block of code:

    Bbasis <- splineDesign(knots = kev[[j]], x = 1:cutoff, ord=4)
    Id <- unique(data$Id)
    kal <- data.frame(do.call("rbind", lapply(1:length(Id), function(i) .wcecalc(ev, data$dose[data$Id==Id[i]],data$Stop[data$Id==Id[i]],Bbasis, cutoff, kev[[j]]))))

    Args:
        ids ((N,) int tensor): the patients ids.
        times ((N,) int tensor): observation times.
        doses ((N,) tensor): the doses received.
        knots ((K,) tensor): the positions of the knots.
        cutoff (int): the length of the observation window.
        order (int): the order of the B-Spline basis.

    Returns:
        tuple of ((N, F) tensor, (K,) tensor):
            - Values of the F B-Spline atom functions, sampled at the required 
              observation times.
            - Positions of the knots.
    """

    # Step 1: Pre-processing ===================================================

    # 1.a: Basic re-ordering and counting --------------------------------------

    # Extract the main integer dimensions:
    (N,) = ids.shape  # Number of samples, number of input features

    # Re-order the input variables to make sure that the patient ids are contiguous
    # in memory. This is necessary to ensure fast batch processing with KeOps:
    sort_ids = ids.argsort()

    sorted_ids = ids[sort_ids]  # (N,), e.g. [0, 0, 0, 0, 1, 1, 2, 2, 2, 3]
    times = times[sort_ids]  #    (N,), e.g. [3, 5, 6, 8, 2, 5, 5, 6, 7, 6]
    doses = doses[sort_ids]  #    (N,), e.g. [.1,0, 0,.8, 0, 4, 0, 5, 0, 0]

    # In the "sources" for the BSpline convolution, discard all the times
    # where the drug has not been taken:
    mask = doses != 0  # (N,) vector of bool
    source_weights = doses[mask]  # (M,)
    source_times = times[mask]  # (M,)
    source_ids = sorted_ids[mask]  # (M,)

    # 1.b: create the knots and cutoff window ----------------------------------

    # Use quantiles for knots placement:
    if knots is None:
        knots = place_knots(cutoff=cutoff, nknots=nknots, order=order)
        knots = torch.tensor(knots, device=device, dtype=float32)

    # The window is a (2,) tensor:
    window = torch.tensor([1.0, cutoff + 1.0], device=device, dtype=float32)

    # Step 2: actual computation ===============================================
    # features_i is a (N,F) tensor:
    features_i = bspline_conv(
        target_times=times,
        target_ids=sorted_ids,
        source_times=source_times,
        source_weights=source_weights,
        source_ids=source_ids,
        knots=knots,
        window=window,
        order=order,
    )

    # Step 3: roll-back the re-ordering of Step 1 ==============================

    features = torch.zeros_like(features_i)
    # equivalent to "features = features_i[inverse(sort_ids)]":
    features[sort_ids] = features_i
    return features, knots


def bspline_atoms(*, cutoff, nknots=1, order=3, knots=None):
    """Returns a set of B-Spline functions sampled on [0, cutoff-1].

    The number of B-spline covariates is equal to
     F = (K - order - 1)
    where K is the length of `knots` if it is not None,
    or is equal to (nknots + 2 + 2*order) otherwise (leading to F = nknots+order+1).

    Args:
        cutoff (int): size of the time window.
        nknots (int, optional): number of inner knots. Defaults to 1.
        order (int, optional): order of the B-Splines. Defaults to 3 (=cubic splines).
        knots ((K,) tensor, optional): specific values for the B-Spline knots,
            to use instead of relying on nknots. Defaults to None.

    Returns:
        tuple of ((cutoff, F) tensor, (K,) tensor):
            - Values of the F B-Spline atom functions, sampled on [0, 1, ..., cutoff-1].
            - Positions of the knots.
    """

    times = torch.arange(0, cutoff, device=device, dtype=int32)
    N = len(times)

    # Dummy vector of "ids" (we create one patient only):
    ids = torch.zeros(N, device=device, dtype=int32)

    # Doses:
    doses = torch.zeros(N, device=device, dtype=float32)
    doses[times == 0] = 1

    features, knots = wce_features_batch(
        ids=ids,
        times=times,
        doses=doses,
        nknots=nknots,
        cutoff=cutoff,
        order=order,
        knots=knots,
    )

    return features, knots
