from survivalgpu.wce import wce_torch
from survivalgpu.simulation import simulate_dataset

import torch

def test():
    print()
    n_patients = 1000
    max_time = 365
    HR_target = 2.8
    doses = [1,1.5,2,2.5,3]
    cutoff = 180
    scenario = "exponential_scenario"
    dataset = simulate_dataset(max_time = max_time, n_patients = n_patients, doses = doses, scenario = scenario, cutoff = cutoff, HR_target = HR_target)

    ids = torch.tensor(dataset["patient"])
    doses = torch.tensor(dataset["dose"])
    events = torch.tensor(dataset["event"])
    times = torch.tensor(dataset["stop"])


    print(ids)

    model = wce_torch(
        covariates=None,
        ids = ids, 
        doses = doses,
        events = events, 
        times = times,
        constrained = "Right",
        cutoff = cutoff

    )

    # print(model)
    print(model)


    print("OK")

if __name__ == "__main__":
    test()


# n_patients = 10
# max_time = 365
# cutoff = 180
# 
# ]
# 



# # Xmat = generate_Xmat(max_time,n_patients,[1,2,3])

# wce_mat = simulate_dataset(max_time = max_time, n_patients = n_patients, doses = doses, scenario = scenario, cutoff = cutoff, HR_target = HR_target)


