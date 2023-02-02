# %%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from IPython.display import display
import torch
from pykeops.torch import LazyTensor

# Notebook settings
pd.set_option("max_rows", 5)
pd.set_option("precision", 2)
DISPLAY = True  # Display figures
TABLES = True  # Display tables

FIRST_YEAR = 2008  # First date in the database
START_YEAR = 2009  # (Reference) start date for our study
LAST_YEAR = 2018  # Last date for our study
END_YEAR = 2018  # Last date in the database

INTERESTING_DRUG = "HYDROXYCHLOROQUINE"

# Parameters of our BSpline window:
ORDER = 3  # 3 -> cubic splines
CUTOFF = 365 * 10
# KNOTS = [1] * (ORDER + 1) + [30 * 1, 30 * 2, 30 * 3, 30 * 6, 365, 365 * 2, 365 * 5] + [CUTOFF] * (ORDER + 1)
KNOTS = (
    [1] * (ORDER + 1)
    + [30 * 2, 30 * 3, 30 * 6, 365, 365 * 2, 365 * 5]
    + [CUTOFF] * (ORDER + 1)
)

# %%
# Encoding routines for PyTorch GPU arrays.
use_cuda = torch.cuda.is_available()
tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
inttensor = torch.cuda.IntTensor if use_cuda else torch.IntTensor

to_torch = lambda x: tensor(x.values.astype("float32"))
to_torchint = lambda x: inttensor(x.values.astype("int32"))


def bsplineconv(times_target, times_source, weights_source, ranges=None):
    # N.B.: In the original WCE package, the BSpline basis is created
    #       on a domain x = [1, ..., cutoff] instead of [0, ..., cutoff].
    #       As a consequence, we should offset the event times "myev"
    #       by 1 to retrieve the exact same results.
    events_i = LazyTensor(to_torch(times_target + 1).view(-1, 1, 1))
    doses_times_j = LazyTensor(to_torch(times_source).view(1, -1, 1))
    doses_values_j = LazyTensor(to_torch(weights_source).view(1, -1, 1))

    # The constant parameters are simply encoded as (9,) and (2,) vectors:
    knots_ = LazyTensor(tensor(KNOTS).view(1, 1, -1))

    # Our rule for the "cutoff" window will be to ensure that
    # 1 <= my_ev_i - stop_j < cutoff + 1
    cut_ = LazyTensor(tensor([1.0, CUTOFF + 1.0]).view(1, 1, -1))

    # Symbolic KeOps computation.
    window_ij = cut_.bspline(events_i - doses_times_j, 1)
    atoms_ij = knots_.bspline(events_i - doses_times_j, ORDER + 1)
    full_ij = window_ij * atoms_ij * doses_values_j

    # Block-diagonal ranges, if needed:
    full_ij.ranges = ranges

    # Sum over the source index "j":
    return full_ij.sum(1)


if True or DISPLAY:
    times = np.arange(-2 * 365, 12 * 365)
    atoms = (
        bsplineconv(
            pd.Series(times),
            pd.Series([0.0, 1.0]),
            pd.Series([1.0, 0.0]),
        )
        .cpu()
        .numpy()
    )

    plt.figure(figsize=(16, 5))
    for i, atom in enumerate(atoms.T):
        plt.plot(times / 365, atom, label=f"BSpline {i}")
    plt.plot(times / 365, np.sum(atoms, axis=1), label="Sum")
    plt.xlabel("Years")
    plt.title("BSpline atoms")
    plt.legend()

    plt.figure(figsize=(16, 5))
    for i, atom in enumerate(atoms.T):
        plt.plot(times / 30, atom, label=f"BSpline {i}")
    plt.plot(times / 30, np.sum(atoms, axis=1), label="Sum")
    plt.xlabel("Months")
    plt.xticks(np.arange(13))
    plt.title("BSpline atoms")
    plt.legend(loc="right")
    plt.axis([-0.2, 12, -0.1, 1.1])


# %%
# Test data from Pierre - July 2021
patients = (
    pd.read_csv("patient_VF.csv")
    .drop(["Unnamed: 0"], axis=1)
    .rename(columns={"BEN_SEX_COD": "sex", "AGE_ANN": "age"})
)
# Patients' sexes: 1 = Male, 2 = Female.
patients["sex"] = patients["sex"].map({1: "male", 2: "female"}).astype("category")

# Patients' birthdays: ages are given on the 1st of January, 2009.
patients["birthday"] = pd.Timestamp(f"{START_YEAR}-01-01") - patients[
    "age"
] * pd.offsets.DateOffset(years=1)

# Patients' ranges of observed dates, from January 2009 to December 2018:
patients["first_day"] = (
    pd.Timestamp(f"{START_YEAR}-01-01") - patients["birthday"]
) // pd.Timedelta("1D")
patients["last_day"] = (
    pd.Timestamp(f"{END_YEAR}-12-31") - patients["birthday"]
) // pd.Timedelta("1D")

patients = patients.set_index("id")

if TABLES:
    display(patients)

# %%
# Interesting statistics - distributions of ages and gravity per sex:
if DISPLAY:
    _ = patients.groupby(["sex", "gravity"])["age"].plot.hist(
        histtype="step", title="Age distributions", legend=True, figsize=(8, 4)
    )

# %%
# The drug of interest is stored in a separate table:
precriptions_hydro = pd.read_csv("conso_hydroxychloroquine_VF.csv").rename(
    columns={"PHA_ATC_C07": "variable"}
)
precriptions_hydro["PHA_ATC_L07"] = "HYDROXYCHLOROQUINE"

# Other drugs:
prescriptions = pd.read_csv("prescription_VF.csv")

# We concatenate everything in a large table and rename some columns:
prescriptions = (
    pd.concat([precriptions_hydro, prescriptions])
    .drop(["Unnamed: 0"], axis=1)
    .rename(
        columns={
            "EXE_SOI_DTD": "date",  # Prescription date
            "variable": "ATC",  # ATC label
            "PHA_ATC_L07": "drug",  # Human-readable name
        }
    )
    .reset_index(drop=True)
)

# For performance, drugs are encoded as categories instead of strings:
prescriptions["ATC"] = prescriptions["ATC"].astype("category")
# To keep label shorts, we only keep the first words of drug names:
prescriptions["drug"] = prescriptions["drug"].str.split(" ").str[0].astype("category")
# Use Pandas datetime parser for prescription dates:
prescriptions["date"] = pd.to_datetime(prescriptions["date"], format="%Y-%m-%d")
# Compute the year of prescription:
prescriptions["year"] = prescriptions["date"].dt.year

# For every prescription, retrieve the birthday of the patient:
prescriptions["birthday"] = prescriptions["id"].map(patients["birthday"])
# Some patients cannot be found in the "patients" database:
# we remove their prescriptions from the file.
prescriptions = prescriptions.dropna(axis=0)

# For every single remaning prescription, we know the age (in days)
# of the patient at prescription time:
prescriptions["age_days"] = (
    prescriptions["date"] - prescriptions["birthday"]
) // pd.Timedelta("1D")

# Re-order our columns and drop the (now useless) birthday:
prescriptions = prescriptions[["id", "drug", "ATC", "date", "year", "age_days"]]

if TABLES:
    display(prescriptions)

# %%
# Display the prescription history:
if DISPLAY:
    _ = prescriptions.groupby(["drug"])["year"].plot.hist(
        bins=np.arange(FIRST_YEAR - 1, LAST_YEAR + 6),
        histtype="step",
        title="Drug prescriptions",
        legend=True,
        figsize=(12, 5),
    )

# %%
def duration(x):
    return np.max(x) - np.min(x) + 1


def early_prescription(x):
    return np.min(x) < START_YEAR


def ongoing_prescription(x):
    return np.max(x) >= END_YEAR


# As events (= "death" in survival analysis),
# we take the first prescriptions of our drugs:
events = (
    prescriptions.groupby(["drug", "id"])[["age_days", "year"]]
    .agg(
        year=("year", "min"),  # Year of the first prescription
        days=("age_days", "min"),  # Age (in days) of the first prescription
        duration=(
            "age_days",
            duration,
        ),  # (Lower bound on the) length of the prescription
        total_dose=("age_days", "count"),  # N.B.: All prescriptions count for 1 dose
        early=("year", early_prescription),  # Did the treatment start before 2009?
        ongoing=("year", ongoing_prescription),  # Is the treatment "alive"?
    )
    .dropna()
    .astype("int64")
    .reset_index()
    .pivot(index="id", columns="drug")
    .astype("Int64")
    .swaplevel(axis=1)
    .sort_index(axis=1)
)

# Sort our drugs by prescription rate:
prescription_rate = events.swaplevel(axis=1).sort_index(axis=1)["days"].notna().mean()
sorted_drugs = prescription_rate.sort_values(ascending=False).index
events = events[sorted_drugs]

if TABLES:
    display(events)

# %%
# Display information on our events:
event_days = events.swaplevel(axis=1).sort_index(axis=1)["days"][sorted_drugs]
event_years = event_days / 365

ongoing_status = events.swaplevel(axis=1).sort_index(axis=1)["ongoing"][sorted_drugs]
early_status = events.swaplevel(axis=1).sort_index(axis=1)["early"][sorted_drugs]

if DISPLAY:
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 4))

    _ = pd.DataFrame(
        {
            f"Started < {START_YEAR}, stopped.": (
                (early_status == 1) & (ongoing_status == 0)
            ).mean(),
            f"Started >= {START_YEAR}, stopped.": (
                (early_status == 0) & (ongoing_status == 0)
            ).mean(),
            f"Started < {START_YEAR}, still ongoing in {END_YEAR}.": (
                (early_status == 1) & (ongoing_status == 1)
            ).mean(),
            f"Started >= {START_YEAR}, still ongoing in {END_YEAR}.": (
                (early_status == 0) & (ongoing_status == 1)
            ).mean(),
        }
    ).plot.bar(
        title="Prescription rate",
        stacked=True,
        ax=axes[0],
        colors=["#a29fe3", "#3f38d9", "#e39f9a", "#d94338"],
    )

    _ = event_years.plot.hist(
        bins=np.arange(100),
        histtype="step",
        title="Event ages",
        legend=True,
        ax=axes[1],
    )
    _ = plt.legend(loc="upper left")


# %%
from matplotlib.colors import LinearSegmentedColormap

if DISPLAY:
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(20, 15))
    axes = axes.reshape(-1)

    for i, drug in enumerate(sorted_drugs):
        ev = events[drug]
        axes[i].scatter(
            x=ev["days"] / 365,
            y=ev["duration"] / 365,
            s=ev["total_dose"] / 4,
            c=ev["ongoing"] * 2 + (1 - ev["early"]),
            cmap=LinearSegmentedColormap.from_list(
                "my_br", ["#a29fe3", "#3f38d9", "#e39f9a", "#d94338"]
            ),
        )
        axes[i].set_title(drug)
        axes[i].set_xlabel("start")
        axes[i].set_ylabel("duration")

# %%
# Create a list of all the "interesting ages"
def preprocess(name):
    events_name = (
        events.drop(INTERESTING_DRUG, axis=1).swaplevel(axis=1).sort_index(axis=1)[name]
    )

    events_name.columns = events_name.columns.tolist()
    events_name["id"] = events_name.index

    events_name = events_name.melt(
        id_vars=["id"], var_name="drug", value_name=name
    ).dropna()
    return events_name


events_ages = preprocess("days")
events_early = preprocess("early")

if TABLES:
    display(events_ages)
    display(events_early[events_early["early"] == 0])

events_ages = (
    events_ages[events_early["early"] == 0].sort_values("days").reset_index(drop=True)
)

if False:
    _ = (events_ages["days"] / 365).plot.hist(
        bins=np.arange(100), histtype="step", title="Event ages", legend=True
    )

if TABLES:
    display(events_ages)

# %%
# Add the slice indices with respect to the table of "events_ages":
patients["first_ind"] = events_ages["days"].searchsorted(
    patients["first_day"], side="left"
)
patients["last_ind"] = events_ages["days"].searchsorted(
    patients["last_day"], side="right"
)
# patients["cum_ind_events"] = (patients["last_ind"] - patients["first_ind"]).cumsum()

if True:
    # Check that all events fall within their respective ranges:
    min_ind = events_ages["id"].map(patients["first_ind"])
    max_ind = events_ages["id"].map(patients["last_ind"])
    fits = (min_ind <= events_ages.index) & (events_ages.index < max_ind)
    assert fits.all()

if TABLES:
    display(patients)

# %%
# Table of "id x drug" that records event times
# (or NaN if the drug was never prescribed to a given patient):
events_starts = (
    events.swaplevel(axis=1).sort_index(axis=1)["days"].fillna(np.iinfo(np.int64).max)
)

events_slices, slices_lengths = [], []
for ((index, patient), (index2, event_starts)) in zip(
    patients.iterrows(), events_starts.iterrows()
):
    # Extract all the event times that fall within the subject's "lifetime":
    assert index == index2

    events_slice = (
        events_ages.iloc[patient["first_ind"] : patient["last_ind"]]
        .drop("id", axis=1)
        .drop_duplicates()
    )
    events_slice["target_id"] = index
    # Once the patient has been prescribed a drug, it stops
    # being a "live" subject for this drug.
    # For a given drug and patient, we should thus remove
    # all events that fall after this date:
    events_slice["cutoff"] = events_slice["drug"].map(event_starts)
    events_slice["is_event"] = events_slice["days"] == events_slice["cutoff"]
    events_slice = events_slice[events_slice["days"] <= events_slice["cutoff"]]
    events_slices.append(events_slice)
    slices_lengths.append(len(events_slice))

events_slices = pd.concat(events_slices).reset_index(drop=True)[
    ["target_id", "drug", "days", "is_event"]
]

patients["cum_ind_events"] = np.cumsum(slices_lengths)

if TABLES:
    display(events_slices)

# %%
doses = prescriptions[prescriptions["drug"] == INTERESTING_DRUG]
doses["dose"] = 1  # What matters here is exposition - not the precise dose
doses = doses[["id", "age_days", "dose"]]

n_doses = doses[["id", "dose"]].groupby("id").count()
patients["cum_ind_doses"] = n_doses.cumsum()

if TABLES:
    display(doses)
    display(patients)

# %%
# At this point:
#
# - "events_slices" contains the (stacked) "list of lists" of interesting
#   "event times" per subject.
# - "patients["cum_ind_events"]" contains the indices to read "events_slices".
#
# - "doses" contains the (stacked) "list of lists" of INTERESTING_DRUG doses per subject.
# - "patients["cum_ind_doses"]" contains the indices to read "doses".
#
# We can now perform a KeOps reduction to compute the WCE B-Spline co-variates
# using a block-diagonal reduction.

# Step 1: create a block-diagonal sparsity mask.
def cum_ind_to_ranges(cum_ind):
    """Turns [a, b, ..., y, z] into [[0, a], [a, b], ..., [y, z]]"""
    cum_ind = to_torchint(cum_ind)  # Pandas to NumPy
    cum_ind = torch.cat((inttensor([0]), cum_ind))
    return torch.stack((cum_ind[:-1], cum_ind[1:])).t().contiguous()


ranges_i = cum_ind_to_ranges(patients["cum_ind_events"])
ranges_j = cum_ind_to_ranges(patients["cum_ind_doses"])

# We are now ready to implement our black-diagonal KeOps
# sparsity mask, following the syntax that is detailed at
# https://www.kernel-operations.io/keops/python/sparsity.html.
slices_i = torch.arange(len(ranges_i) + 1)[1:]  # [1, 2, 3, ..., n_Id]

ranges = (
    ranges_i,
    slices_i,
    ranges_j,
    ranges_j,
    slices_i,
    ranges_i,
)
ranges = tuple(x.type(torch.int32).cuda() for x in ranges)

# Step 2: actual computation
covariates_i = bsplineconv(
    events_slices["days"],
    doses["age_days"],
    doses["dose"],
    ranges=ranges,
)
nsplines = covariates_i.shape[1]

# %%
events_covariates = pd.concat(
    (
        events_slices,
        pd.DataFrame(covariates_i.cpu().numpy()),
    ),
    axis=1,
)

if TABLES:
    display(events_covariates)

if False and DISPLAY:
    myid = 1
    mydoses = doses[doses["id"] == myid]
    myevents = events_covariates[events_covariates["target_id"] == myid]
    plt.figure(figsize=(16, 5))
    plt.plot(myevents["days"] / 365, myevents[0])
    plt.scatter(mydoses["age_days"] / 365, 0 * mydoses["dose"])
    plt.axis([53, 54, 0, 1])

# %%
TEST_DRUG = "AMOXICILLINE"

sorted_covariates = events_covariates.sort_values(["drug", "days", "is_event"])
sorted_covariates = sorted_covariates[
    ["drug", "days", "is_event", "target_id"] + list(range(nsplines))
]

sorted_covariates = sorted_covariates[
    sorted_covariates["drug"] == TEST_DRUG
].reset_index(drop=True)


def is_event(x):
    return (x == True).sum()


def not_event(x):
    return (x == False).sum()


events_info = sorted_covariates.groupby("days").agg(
    is_event=("is_event", is_event),
    not_event=("is_event", not_event),
)


if TABLES:
    display(sorted_covariates)
    display(events_info)

if DISPLAY:
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 4))

    events_info["is_event"].plot.hist(
        histtype="step",
        title='"Active" patients per event time',
        legend=True,
        bins=np.arange(events_info["is_event"].max() + 2) - 0.5,
        ax=axes[0],
    )

    events_info["not_event"].plot.hist(
        histtype="step",
        title='"Other" patients per event time',
        legend=True,
        ax=axes[1],
    )

# %%
if TABLES:
    display(events_info[events_info["is_event"] == 2])
    with pd.option_context(
        "display.max_rows", 8, "display.max_columns", None
    ):  # more options can be specified also
        display(sorted_covariates[sorted_covariates["days"] == 32812])

# %%
covariates_events = sorted_covariates[sorted_covariates["is_event"]]
covariates_other = sorted_covariates[~sorted_covariates["is_event"]]

if TABLES:
    display(events_info.sum())
    display(covariates_events)
    display(covariates_other)

# %%
# We simulate bootstrapping using an integer array
# of "copies" numbers of shape (n_patients, n_bootstraps).
# The original sample corresponds to copies = [1, ..., 1],
# while other values for the vector always sum up to
# the number of patients.
n_patients = len(patients)  # Number of patients
BOOTSTRAPS = 1  # Acts as a batch dimension
BATCHSIZE = 1  # Divider of BOOTSTRAPS, used to bypass register spilling

bootstrap_indices = torch.randint(n_patients, (BOOTSTRAPS, n_patients)).type(inttensor)
# Our first column corresponds to the original sample:
bootstrap_indices[0, :] = torch.arange(n_patients)

copies = torch.stack(
    [torch.bincount(b_ind, minlength=n_patients) for b_ind in bootstrap_indices]
)
copies = copies.type(tensor) / n_patients

if TABLES:
    display(copies)
    display(copies.shape)

if DISPLAY:
    plt.figure(figsize=(20, 4))
    plt.title("Distributions of our bootstrap samples")
    for copy in copies:
        plt.step(np.arange(n_patients), copy.cpu().numpy())

# %%
# Clustering information:
info = to_torchint(events_info)
n_times = len(info)


def cum_ind_to_ranges(cum_ind):
    """Turns [a, b, ..., y, z] into [[0, a], [a, b], ..., [y, z]]"""
    cum_ind = cum_ind.type(inttensor)
    cum_ind = torch.cat((inttensor([0]), cum_ind))
    return torch.stack((cum_ind[:-1], cum_ind[1:])).t().contiguous()


cast_ranges = lambda r: tuple(x.type(torch.int32).cuda() for x in r)

# Info to build "skinny" block-diagonal ranges:
slices_times = torch.arange(n_times + 1)[1:]  # [1, 2, 3, ..., n_Id]
times_ranges = cum_ind_to_ranges(slices_times)  # [[0, 1], [1, 2], ..., [n_Id-1, n_Id]]


# Convert our dataframes to torch "dicts"
def pandas_to_torch(ev, info_column):
    torch_ev = {
        "days": to_torchint(ev["days"]),
        "id": to_torchint(ev["target_id"]),
        "covariates": to_torch(ev[list(range(nsplines))]),
    }
    # For every subject, fetch the (BOOTSTRAPS,) vector of "n_copies":
    torch_ev["copies"] = copies.t()[torch_ev["id"].long(), :].type(tensor)

    # Create the block-sparsity ranges for the clustering:
    subjects_per_time = info[:, info_column]
    torch_ev["ranges"] = cum_ind_to_ranges(subjects_per_time.cumsum(dim=0))

    # Compute the numbers of copies per time with
    # a skinny block-sparse reduction:
    time_copies = torch.zeros(n_times, BOOTSTRAPS).type(tensor)
    ranges = cast_ranges(
        (
            times_ranges,
            slices_times,
            torch_ev["ranges"],
            torch_ev["ranges"],
            slices_times,
            times_ranges,
        )
    )

    for start in range(0, BOOTSTRAPS, BATCHSIZE):
        end = start + BATCHSIZE
        copies_j = LazyTensor(torch_ev["copies"][None, :, start:end].contiguous())
        # Small hack so that KeOps is aware that we need n_times lines...
        zeroes_i = LazyTensor(torch.zeros(n_times).type(tensor).view(n_times, 1, 1))
        copies_j = copies_j + zeroes_i
        # Skinny block-sparsity range:
        copies_j.ranges = ranges
        time_copies[:, start:end] = copies_j.sum(1)
    torch_ev["time_copies"] = time_copies

    return torch_ev


deads = pandas_to_torch(covariates_events, 0)
survivors = pandas_to_torch(covariates_other, 1)


if False and TABLES:
    display(deads)
    display(survivors)

# %%
def loglikelihood_deads(w):
    """w is a (n_covariates, n_bootstraps) matrix."""
    scores = deads["covariates"] @ w  # (n_deads, n_bootstraps)
    return (scores * deads["copies"]).sum()


# %%
n_survivors = survivors["covariates"].shape[0]
logweight = (survivors["copies"] + 1e-8).log()  # (n_survivors, n_bootstraps)
lognorm = (survivors["time_copies"] + 1e-8).log()  # (n_times, n_bootstraps)


def loglikelihood_survivors(w, eps=1):
    """w is a (n_covariates, n_bootstraps) matrix."""
    scores = survivors["covariates"] @ w  # (n_survivors, n_bootstraps)
    scores = scores / eps

    # print(scores)

    ranges = cast_ranges(
        (
            times_ranges,
            slices_times,
            survivors["ranges"],
            survivors["ranges"],
            slices_times,
            times_ranges,
        )
    )

    full_loss = 0
    batchsize = 1
    for start in range(0, BOOTSTRAPS, batchsize):
        end = start + batchsize
        scores_j = LazyTensor(
            scores[:, start:end].contiguous().view(1, n_survivors, batchsize)
        )
        logweight_j = LazyTensor(
            logweight[:, start:end].contiguous().view(1, n_survivors, batchsize)
        )
        lognorm_i = LazyTensor(
            lognorm[:, start:end].contiguous().view(n_times, 1, batchsize)
        )
        softmin_ij = (
            scores_j + logweight_j - lognorm_i
        )  # (n_times, n_survivors, BATCHSIZE)
        softmin_ij.ranges = ranges
        softmin = eps * softmin_ij.logsumexp(dim=1)  # (n_times, BATCHSIZE)

        dead_copies = deads["time_copies"][
            :, start:end
        ].contiguous()  # (n_times, BATCHSIZE)
        full_loss = full_loss - (dead_copies * softmin).sum()

    return full_loss


# %%
from tqdm import tqdm

# The parameters of our model:
n_covariates = deads["covariates"].shape[1]
weights = torch.zeros((n_covariates, BOOTSTRAPS)).type(tensor)
weights.requires_grad = True

# Optimization loop
n_its = 10
# optimizer = torch.optim.Adam([weights], lr = 1)
optimizer = torch.optim.LBFGS([weights], line_search_fn="strong_wolfe")
# optimizer = torch.optim.SGD([weights], lr=.1, momentum=.9)


def closure():
    optimizer.zero_grad()
    loss = -(loglikelihood_deads(weights) + loglikelihood_survivors(weights))
    loss.backward()
    return loss


for t in tqdm(range(n_its), ncols=100):
    optimizer.step(closure)


# %%
print(loglikelihood_deads(weights))
print(loglikelihood_survivors(weights))

# %%
print(weights)
print(weights.grad)

# %%
if True or DISPLAY:
    risks = atoms @ weights.detach().cpu().numpy()

    plt.figure(figsize=(16, 5))
    for i, risk in enumerate(risks.T):
        plt.plot(times / 365, risk, label=f"Risk function {i}")

    plt.xlabel("Years")
    plt.title(f"Risk functions for {TEST_DRUG}")
    plt.legend()

    plt.figure(figsize=(16, 5))
    for i, risk in enumerate(risks.T):
        plt.plot(times / 30, risk, label=f"Risk function  {i}")
    plt.xlabel("Months")
    plt.xticks(np.arange(13))
    plt.title(f"Risk functions for {TEST_DRUG}")
    plt.legend(loc="right")
    plt.axis([-0.2, 12, -1.1, 1.1])
