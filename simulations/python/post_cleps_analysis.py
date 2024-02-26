import csv
import torch 
import sys
import matplotlib.pyplot as plt
import numpy as np
import os

from pykeops.torch import LazyTensor



sys.path.append("../../python")
from survivalgpu import use_cuda, device, float32, int32, int64
from survivalgpu.utils import numpy
from survivalgpu import wce_torch
from survivalgpu.wce_features import wce_features_batch


simualtion = "16-01-2024"
path =  "../Simulation_results/"+ simualtion + "/"
print(path)


list_dir = os.listdir(path)
print(list_dir)

weight_function = "exponential_weight"


 

for file_name in list_dir:


    file_path =  "../Simulation_results/"+ simualtion + "/" + file_name

    data = torch.load(file_path, map_location=torch.device('cpu'))


    weight_function_path = "../weight_functions_shapes/" + weight_function + ".csv"


    with open (weight_function_path) as file:
        rows = []
        csvreader = csv.reader(file)
        next(csvreader)
        for row in csvreader:
            rows.append(float(row[0]))

    rows = rows[1:]
    # print(rows)

    # plt.plot(rows)

    mean_tensor = data["WCEmat"].mean(dim=0)
    std_tensor = data["WCEmat"].std(dim=0)

    

    # print(mean_tensor)
    # print(mean_tensor + std_tensor)
    # print(mean_tensor - std_tensor)

    
    plt.plot(mean_tensor.tolist(),color = "black", linewidth =0.85)
    plt.plot((mean_tensor + std_tensor).tolist(), color = "black", linestyle = "dashed", linewidth =0.85)
    plt.plot((mean_tensor - std_tensor).tolist(), color = "black",linestyle = "dashed", linewidth =0.85)
    plt.plot(rows,color = "red", linewidth =1)
    plt.title(file_name)
    plt.xlabel("time (days)")
    plt.ylabel("weight")
    figure_path = path + "/results/" + file_name + ".png"
    plt.savefig(figure_path)

    print (figure_path)
    plt.clf()
