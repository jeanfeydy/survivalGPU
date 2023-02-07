# Standard imports:
from tqdm import tqdm
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.profiler import profile, record_function, ProfilerActivity

from tests.utils import numpy, form, wce_features, coxph_fit
from survivalgpu.datasets import drug_dataset


# Setup ==================================================================================
use_cuda = torch.cuda.is_available()

ds = drug_dataset(
    Drugs=4,
    Patients=100,
    Times=100,
    device="cuda" if use_cuda else "cpu",
)

# WCE Analysis ===========================================================================
# Features = B-spline covariates: --------------------------------------------------------
# We compute in parallel all the (Drugs, Patients, Times, Features) values

ds_wce = wce_features(
    doses=ds["doses"],
    times=ds["times"],
    nknots=1,
    order=3,
    cutoff=10,
    constrained=None,
)

ds = {**ds, **ds_wce}

print("B-Spline knots:", form(ds["knots"]))
print("B-Spline function areas:", form(ds["areas"]))


# Compute the "true" output of the CoxPH model: --------------------------------------------
model = coxph_fit(
    exposures=ds["exposures"],
    times=ds["times"],
    events=ds["events"],
    areas=ds["areas"],
)

# Perform the permutation test: -----------------------------------------------------------
Permutations, Bootstraps = 100, 1000


def permutation_test(*, exposures, times, events, areas, Permutations):

    Drugs, Patients, Times, Features = exposures.shape
    permutation_risks = torch.zeros(Permutations, Drugs, device=exposures.device)

    # with torch.autograd.profiler.profile(use_cuda=use_cuda) as prof:
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    # with profile(activities=[ProfilerActivity.CPU]) as prof:
    for p in tqdm(range(Permutations)):
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
    assert permutation_risks.shape == (Permutations, Drugs)
    return permutation_risks


# permutation_risks is a (Permutations, Drugs) Tensor that contains the estimated total risk
# associated to each drug - i.e. the total area under the risk function - for a control
# case, where the relationship between the covariates and the events has been destroyed.
permutation_risks = permutation_test(
    exposures=ds["exposures"],
    times=ds["times"],
    events=ds["events"],
    areas=ds["areas"],
    Permutations=100,
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

# Our main run was in no batch mode, so we "pop" the first dimension
stds = coxph_output["imat"][0].diagonal(dim1=-2, dim2=-1).sqrt()
Hessian = coxph_output["hessian"][0]
Imat = coxph_output["imat"][0]

assert stds.shape == (Drugs, Features)
assert Hessian.shape == (Drugs, Features, Features)
assert Imat.shape == (Drugs, Features, Features)


# Rough Gaussian-like esimation of the confidence intervals for the total risk area: -----
# We use a simple model:
# the ideal coefs are the minimizers of the neglog-likelihood function of the CoxPH model,
# with gradient = 0 at the optimum and a Hessian that is a positive-definite matrix
# of shape (Features, Features) for each drug.
# For each drug, we may reasonably expect the "coefs" vector to follow a Gaussian
# distribution with:
# - mean = estimated vector "coefs[drug]"
# - covariance = Imat[drug] = inverse(Hessian[drug]).
#
# In this context, the total risk area = \sum_{b-spline atom i} areas[i] * coef[i]
# is a 1D-Gaussian vector with:
# - mean[drug] = \sum_{b-spline atom i} areas[i] * estimated_coefs[drug,i]
# - variance[drug] = \sum_{i, j} areas[i] * areas[j] *

risk_variances = torch.einsum("ijk,j,k->i", Imat, areas, areas)
risk_stds = risk_variances.sqrt()
print("Estimated log-HR:", form(risk_means), "+-", form(risk_stds))

assert risk_variances.shape == (Drugs,)
assert risk_stds.shape == (Drugs,)

# ci_95 = 1.96 / np.sqrt(coefs.shape[-1])

# (Drugs, Features, Features) @ (Features,)
ci_95 = Imat @ areas
ci_95 = 1.96 * ci_95 / risk_stds.view(Drugs, 1)
assert ci_95.shape == (Drugs, Features)

if False:
    print("Area deltas for the 95% CI:")
    print(form(ci_95 @ areas))
    print("Expected values:")
    print(form(1.96 * risk_stds))

x = np.arange(cutoff)

plt.figure()
plt.title("B-Spline atoms")
for i, f in enumerate(atoms.t()):
    plt.plot(numpy(f), label=f"{i}")
plt.legend()
plt.savefig("output_atoms.png")


plt.figure()
plt.title("Estimated risk functions, with 95% CI for the total risk area")
for i, (coef, ci) in enumerate(zip(coefs, ci_95)):
    plt.plot(numpy(atoms @ coef), label=f"{i}")
    plt.fill_between(
        x, numpy(atoms @ (coef - ci)), numpy(atoms @ (coef + ci)), alpha=0.2
    )
plt.legend()
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
plt.title(f"Distribution of the total risk for drug {D}")
t = np.linspace(bootstrap_risk.min().item(), bootstrap_risk.max().item(), 100)
plt.plot(
    t,
    np.exp(-0.5 * (t - risk_mean_est) ** 2 / risk_std_est**2)
    / np.sqrt(2 * np.pi * risk_std_est**2),
    label="Estimation",
)
plt.hist(
    numpy(bootstrap_risk),
    density=True,
    histtype="step",
    bins=50,
    log=True,
    label="Bootstrap",
)
plt.legend()
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
