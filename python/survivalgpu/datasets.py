import torch

# Virtual dataset ========================================================================

# Random doses, with a simple risk model: -----------------------------------------------
def drug_dataset(*, Drugs=1, Patients=1, Times=1, device="cpu"):
    """Create a virtual dataset for testing using a simple risk model.

    Args:
        Drugs (int, optional): number of drugs to test. Defaults to 1.
        Patients (int, optional): number of patients. Defaults to 1.
        Times (int, optional): number of sampling times. Defaults to 1.
        device (str, optional): device where the data should be stored. Defaults to "cpu".

    Returns:
        dict: contains arrays that describe a drug consumption dataset with:
            - "doses": a (Drugs, Patients, Times) float32 Tensor.
            - "times": a (Times,) int32 Tensor with the sampling times.
            - "events": a (Patients, Times) int32 Tensor whose values are equal to
                0 if everything is all right, i.e. the patient is still "alive",
                1 if the patient is "dying" at this exact moment,
                2+ if this consumption happened after the event of interest,
                   and should therefore be removed from the survival analysis.

              Note that we deliberately include drug consumption data "after death"
              as these may be relevant to a permutation test.
    """

    # Sampling times = [0, 1, ..., Times-1]:
    times = torch.arange(Times, dtype=torch.int32, device=device)

    # Random doses - we could try more interesting models:
    doses = 1.0 * (torch.rand(Drugs, Patients, Times, device=device) < 0.5)
    assert doses.shape == (Drugs, Patients, Times)

    # events = 0  if everything is all right,
    #          1  if the patient is dying at this exact moment
    #          2+ if it is too late
    events = torch.zeros(Patients, Times, dtype=torch.int32, device=device)

    # Simple risk model: two consecutive doses of drug 0 -> 80% death.
    danger = doses[0, :, :]  # (Patients, Times)
    events[:, 2:] = torch.logical_and(danger[:, :-2] > 0.5, danger[:, 1:-1] > 0.5)
    events[:] = torch.logical_and(
        events, torch.rand(Patients, Times, device=device) > 0.2
    )
    # Event = 1 only for the first time.
    # All subsequent cells have value >= 2 and should be discarded
    events[:, 1:] += 2 * events[:, :-1].cumsum(axis=-1)
    assert events.shape == (Patients, Times)

    return {
        "doses": doses,
        "times": times,
        "events": events,
    }
