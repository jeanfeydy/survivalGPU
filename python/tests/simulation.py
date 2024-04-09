from survivalgpu.wce import wce_torch
from survivalgpu.simulation import simulate_dataset, get_dataset
import pandas as pd
import numpy as np

import torch

def test():
    print()
    n_patients = 1000
    max_time = 365
    HR_target = 2.8
    doses = [1,1.5,2,2.5,3]
    cutoff = 180
    scenario = "exponential_scenario"
    # dataset = simulate_dataset(max_time = max_time, n_patients = n_patients, doses = doses, scenario = scenario, cutoff = cutoff, HR_target = HR_target)

    # ids = torch.tensor(dataset["patient"])
    # doses = torch.tensor(dataset["dose"])
    # events = torch.tensor(dataset["event"])
    # times = torch.tensor(dataset["stop"])


    # print(ids)

    # model = wce_torch(
    #     covariates=None,
    #     ids = ids, 
    #     doses = doses,
    #     events = events, 
    #     times = times,
    #     constrained = "Right",
    #     cutoff = cutoff

    # )

    # # print(model)
    # print(model)


    print("OK")


    Xmat = np.array([
        [1,2,3],
        [4,5,6],
        [7,8,9],
    ])

    print(Xmat)

    
    wce_id = np.array([1,0,2])
    fup = np.array([2,3,1])
    event = np.array([0,1,1])


    wce_df_true = pd.DataFrame()
    wce_df_true["patient"] = [1,1,2,2,2,3]
    wce_df_true["start"] = [0,1,0,1,2,0]
    wce_df_true["stop"] = [1,2,1,2,3,1]
    wce_df_true["event"] = [0,0,0,0,1,1]
    wce_df_true["dose"] =  [2,5,1,4,7,3]

    print(wce_df_true)

    df_dataset_program = get_dataset(Xmat,3,3,2.8,fup,event,wce_id)
    print(df_dataset_program)


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


