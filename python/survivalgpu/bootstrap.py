import torch
from .typecheck import typecheck, Float32Tensor, Int64Tensor
from .group_reduction import group_reduce


class Bootstrap:
    @typecheck
    def __init__(
        self,
        *,
        indices: Int64Tensor["bootstraps patients"],
        event: Int64Tensor["intervals"],
        patient: Int64Tensor["intervals"],
        group: Int64Tensor["intervals"],
        verbosity: int = 0,
    ):
        """
        We perform efficient bootstrapping using (B, I) arrays of "weights"
        where B is the number of bootstraps and I is the number of intervals.
        The original sample corresponds to weights = [1, ..., 1],
        while other values typically include 0's (for samples that are not drawn)
        and higher integer values (for samples that are drawn several times).

        This object computes all the relevant groups and weights from the following inputs:
        - indices, a (B, P) int64 Tensor
          where B is the number of bootstraps and P is the number of patients.
          Typical choices are:
          - indices = torch.arange(1, P) = [[0, 1, 2, 3, ..., P-1]],
            which corresponds to the original sample.
          - indices = torch.randint(P, (B, P)),
            which corresponds to a random bootstrap sample, e.g. 
            [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  -> "miracle", we retrieve the original sample!
             [3, 3, 1, 2, 8, 6, 0, 0, 8, 9]]  -> "genuinely random" bootstrap sample
          Note that these indices are usually drawn using a random number generator
          that respects the batch and stratification constraints.
          For instance, with P=10 as above, if:
          - batch  = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
          - strata = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
          This defines three groups that should not mix in the bootstrap samples:
          - group 0: [0, 1, 2]
          - group 1: [3, 4, 5, 6]
          - group 2: [7, 8, 9]
          Consequently, with B=2 bootstraps, typical indices would be:
          - indices = [[0, 1, 2 ; 3, 4, 5, 6 ; 7, 8, 9],  -> "miracle", we retrieve the original sample!
                       [0, 0, 1 ; 4, 6, 4, 4 ; 8, 8, 8]]  -> "genuinely random" bootstrap sample

        - event, a (I,) int64 Tensor that indicates the event type for each interval.
        - patient, a (I,) int64 Tensor that indicates the patient for each interval.
        - group, a (I,) int64 Tensor that indicates the group for each interval.
        """
        B, P = indices.shape
        # Step 1: ------------------------------------------------------------------------
        # Compute the numbers of occurrences of each patient index in the
        # rows of bootstrap_indices:
        sample_weights = torch.ones(B, P, dtype=torch.float32, device=indices.device)
        patient_weights = group_reduce(
            values=sample_weights,
            groups=indices,
            reduction="sum",
            output_size=P,
            backend="pyg",
        )
        patient_weights = patient_weights.to(device=indices.device, dtype=torch.float32)
        # Equivalent to:
        # patient_weights 
        # = torch.stack([torch.bincount(b_ind, minlength=P) for b_ind in indices])
        #
        # patient_weights is (B,P) with lines that sum up to P, e.g.
        # [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        #  [2, 1, 1, 2, 0, 0, 1, 0, 2, 1]]

        # Step 2: ------------------------------------------------------------------------
        # Pre-compute the logarithms of the weights:
        # TODO: We are currently adding a small value to prevent NaN.
        #       This is not very clean...
        patient_log_weights = (patient_weights + 1e-8).log()  # (B,P), e.g.
        # [[ 0, 0, 0,  0,   0,   0, 0,    0,  0, 0],
        #  [.7, 0, 0, .7,-inf,-inf, 0, -inf, .7, 0]]

        # Step 3: Aggregate the "total weights for dead samples" at each time point ------
        # These are required as multiplicative factors by the Efron and Breslow rules

        # Compute the total weight of dead samples for every event time:
        dead_weights = weights[:, deaths == 1]
        # dead_weights is (B, Ndeads), e.g.
        # [[1, 1, 1, 1],
        #  [2, 0, 1, 1]]
        dead_cluster_indices = cluster_indices[deaths == 1].repeat(B, 1)
        # dead_cluster_indices is (B, Ndeads), e.g.
        # [[0, 0, 1, 2],
        #  [0, 0, 1, 2]]
        tied_dead_weights = group_reduce(
            values=dead_weights,
            groups=dead_cluster_indices.long(),
            reduction="sum",
            output_size=T,
            backend="pyg",
        )
        # Equivalent to:
        # tied_dead_weights = torch.bincount(cluster_indices[deaths == 1],
        #                     weights=weights.view(-1)[deaths == 1],
        #                     minlength=T)
        #
        # tied_dead_weights is (B,T), e.g.
        # [[2, 1, 1],
        #  [2, 1, 1]]



        if verbosity > 0:
            print("Pre-processing:")
            print(f"Working with {B:,} bootstrap, {C:,} channels, {T:,} death times,")
            print(f"{N:,} rows (= observations) and {D:,} columns (= features).")
            print("")

        if verbosity > 1:
            print("Cluster indices:")
            print(numpy(cluster_indices))
            # print("Number of tied deaths:")
            # print(numpy(tied_deaths))
            print("Tied dead weights:")
            print(numpy(tied_dead_weights))
            print("")
            

for batch_it in range(bootstrap // batchsize):
    # We simulate bootstrapping using an integer array
    # of "weights" numbers of shape (B, N) where B is the number of bootstraps.
    # The original sample corresponds to weights = [1, ..., 1],
    # while other values for the vector always sum up to
    # the number of patients.
    bootstrap_indices = torch.randint(N, (B, N), dtype=int64, device=device)
    # Our first line corresponds to the original sample,
    # i.e. there is no re-sampling if bootstrap == 1:
    if batch_it == 0:
        bootstrap_indices[0, :] = torch.arange(N, dtype=int64, device=device)
    # bootstrap_indices is (B,N), e.g.:
    # [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    #  [3, 3, 1, 2, 8, 6, 0, 0, 8, 9]]

