# Standard imports:
from tqdm import tqdm
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.profiler import profile, record_function, ProfilerActivity

from tests.utils import numpy, form, wce_features, coxph_fit
from survivalgpu.datasets import drug_dataset
from survivalgpu import WCESurvivalAnalysis


# Setup ==================================================================================
use_cuda = torch.cuda.is_available()

# Dict with keys "doses", "times", "events":
ds = drug_dataset(
    drugs=4,
    patients=100,
    times=100,
    device="cuda" if use_cuda else "cpu",
)


# WCE Analysis ===========================================================================

model = WCESurvivalAnalysis(
    nknots=1,
    order=3,
    cutoff=10,
    constrained=None,
)

model.fit(doses=ds["doses"], times=ds["times"], events=ds["events"])

print("B-Spline knots:", form(model.knots))
print("B-Spline function areas:", form(model.atom_areas))


# Perform the permutation test: -----------------------------------------------------------
Permutations, Bootstraps = 100, 1000


def permutation_test(*, exposures, times, events, areas, permutations):

    Drugs, Patients, Times, Features = exposures.shape
    permutation_risks = torch.zeros(permutations, Drugs, device=exposures.device)

    # with torch.autograd.profiler.profile(use_cuda=use_cuda) as prof:
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    # with profile(activities=[ProfilerActivity.CPU]) as prof:
    for p in tqdm(range(permutations)):
        coxph_output_p = coxph_fit(
            exposures=exposures,
            times=times,
            events=events,
            areas=areas,
            permutation=torch.randperm(Patients, device=exposures.device),
        )
        # We remove the dummy batch dimension:
        permutation_risks[p, :] = coxph_output_p["risk"][0]

    # prof.export_chrome_trace("trace_profile_bootstrap.json")
    assert permutation_risks.shape == (permutations, Drugs)
    return permutation_risks


# permutation_risks is a (Permutations, Drugs) Tensor that contains the estimated total risk
# associated to each drug - i.e. the total area under the risk function - for a control
# case, where the relationship between the covariates and the events has been destroyed.
permutation_risks = permutation_test(
    exposures=ds["exposures"],
    times=ds["times"],
    events=ds["events"],
    areas=ds["areas"],
    permutations=100,
)


def cumulative_distributions(x):
    """Compute the cumulative distribution function of the given data"""
    sorted_x, indices = x.sort()

    return x, torch.arange(1, len(x) + 1) / len(x)



# Our main run was in no batch mode, so we "pop" the first dimension:
coefs = model["coef"][0]  # (Drugs, Features)
risk_means = model["risk"][0]  # (Drugs,)


# Compute the normalized rank of each
all_risks = torch.cat((risk_means.view(1, Drugs), permutation_risks), dim=0)
sorted_risks, indices = all_risks.sort(dim=0)

# Invert the lists of indices, drug-wise:
ranks = torch.empty_like(indices)
for d in range(Drugs):
    ranks[indices[:, d], d] = torch.arange(Permutations + 1, device=device)


# The main purpose of a permutation test over a Bonferroni correction
# is that it automatically controls for the number of degrees of freedom
# via the independence (or lack thereof) of the columns of the ranks matrix
max_ranks = ranks.max(dim=-1).values

# Turn the ranks into p-values adjusted for multiple testing:


max_ranks = max_ranks / Permutations
ranks = ranks / Permutations
print("Percentile rank of the Hazard Ratios, for each drug:", form(ranks[0]))
print(
    "Adjusted p-values of the Hazard Rat., for each drug:",
)


# Inspect the results ====================================================================
x = np.arange(cutoff)

plt.figure()
model.display_atoms()
plt.savefig("output_atoms.png")


plt.figure()
model.display_risk_functions()
plt.savefig("output_functions.png")


# Check our estimation against a bootstrap sample for drug D: ============================
D = 0
bootstrap_output = fit_model(
    exposures=exposures,
    events=events,
    times=times,
    bootstrap=Bootstraps,
    interest_drug=D,
)

coef, ci = coefs[D], ci_95[D]
bootstrap_coef = bootstrap_output["coef"]
# (B,) = (B,Features) @ (Features)
bootstrap_risk = bootstrap_output["coef"] @ areas
assert bootstrap_coef.shape == (Bootstraps, Features)

print("Coef - estimation:", form(coef), "+-", form(ci))
print(
    "     - bootstrap :",
    form(bootstrap_coef.mean(axis=0)),
    "+-",
    form(bootstrap_coef.std(axis=0, unbiased=True)),
)

risk_mean_est = risk_means[D].item()
risk_std_est = risk_stds[D].item()
print(f"Total risk - estimation: {risk_mean_est:.3f} +- {risk_std_est:.3f}")
print(
    f"Total risk - bootstrap : {bootstrap_risk.mean(axis=0).item():.3f} +- {bootstrap_risk.std(axis=0, unbiased=True).item():.3f}"
)

plt.figure()
plt.title(f"Bootstrap sample for drug {D}")
for coef in bootstrap_output["coef"]:
    plt.plot(x, numpy(atoms @ coef), color="b", alpha=0.5 / np.sqrt(Bootstraps))
plt.fill_between(
    x, numpy(atoms @ (coef - ci)), numpy(atoms @ (coef + ci)), color="c", alpha=0.5
)
plt.plot(numpy(atoms @ coef), color="k")
plt.savefig("output_bootstrap.png")


plt.figure()
model.display_risk_distribution(drug=0)
plt.savefig("output_bootstrap_risks.png")


plt.figure()
plt.title(
    f"Distribution of the max percentile ranks \nover {Drugs} drugs in the permutation test"
)
plt.hist(
    numpy(max_ranks),
    density=True,
    histtype="step",
)
plt.savefig("output_permutation_max_ranks.png")
