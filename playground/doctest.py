import torch
import os
import sys
import numpy as np


sys.path.append("../python/")
from survivalgpu.simulation import simulate_dataset, WCECovariate, ConstantCovariate, TimeDependentCovariate
from survivalgpu.coxph import coxph_R
from survivalgpu.wce import wce_R
print("Hello World")

Constant_cox = ConstantCovariate(name = "Constant_cox", 
                                 values = [0,1],
                                 coef = np.log(1.5),
                                 weights=[1,2])

list_covariates = [Constant_cox]


dataset = simulate_dataset(max_time =365,
                 n_patients = 1000,
                 list_covariates=list_covariates)

model = coxph_R(dataset,
               stop = "stop",
               death = "events",
               covars=["Constant_cox"],
               )

print(model)
