import csv
import torch 
import sys
import matplotlib.pyplot as plt
import numpy as np
import os
from statistics import mean
from pykeops.torch import LazyTensor



sys.path.append("../../python")
from survivalgpu import use_cuda, device, float32, int32, int64
from survivalgpu.utils import numpy
from survivalgpu import wce_torch
from survivalgpu.wce_features import wce_features_batch


simualtion = "16-01-2024"
path =  "../Simulation_results/"+ simualtion + "/"
# print(path)


list_dir = os.listdir(path)
# print(list_dir)

weight_function = "exponential_weight"


 

for i in range(1):

    file_name = list_dir[0]
    file_path =  "../Simulation_results/"+ simualtion + "/" + file_name

    data = torch.load(file_path, map_location=torch.device('cpu'))


    weight_function_path = "../weight_functions_shapes/" + weight_function + ".csv"




    with open (weight_function_path) as file:
        real_weights = []
        csvreader = csv.reader(file)
        next(csvreader)
        for row in csvreader:
            real_weights.append(float(row[0]))


    real_weights = torch.tensor(real_weights)
    mean_tensor = data["WCEmat"].mean(dim=0)

    mean_tensor = data["WCEmat"].mean(dim=0)
    std_tensor = data["WCEmat"].std(dim=0)

    weights_diff = []

    list_air_diff = []

    for i in range(int(data["WCEmat"].shape[0]/2)):
        WV = abs(data["WCEmat"][i] - real_weights)
        weights_diff.append(WV.mean().item())

        sum_air = data["WCEmat"][i].sum().item()
        sum_air_real = real_weights.sum().item()
        #print(sum_air_real)
        #print(sum_air) ### Ai ai ai sensé être à 1
        air_diff = abs(sum_air_real-sum_air)
        list_air_diff.append(air_diff)
    print(air_diff)
    print(mean(weights_diff))





    # problem of cutoff, will show better eman for bigger cutoff as the curve will 
    # have no differnce when reach 0 
    # get the mean of each datapoint difference for each bootstraps 
    # do the 95% confidence interval 
        
    # Second method 
    # do tha air behind the curve 
    
    # air = 0
    # for i in range(data["WCEmat"].shape[0]):
    # this one is NOT affected by a bigger 


    # L'air sous la courba a une plus grosse différence que la moyenne des différences, peut être du à l'effet 
    # de la queue de la courbe 
    # quels sont les méthdoes statistiques ? 

 



  

    

    

    # print(mean_tensor)
    # print(mean_tensor + std_tensor)
    # print(mean_tensor - std_tensor)

    
    # plt.plot(mean_tensor.tolist(),color = "black", linewidth =0.85)
    # plt.plot((mean_tensor + std_tensor).tolist(), color = "black", linestyle = "dashed", linewidth =0.85)
    # plt.plot((mean_tensor - std_tensor).tolist(), color = "black",linestyle = "dashed", linewidth =0.85)
    # plt.plot(rows,color = "red", linewidth =1)
    # plt.title(file_name)
    # plt.xlabel("time (days)")
    # plt.ylabel("weight")
    # figure_path = path + "/results/" + file_name + ".png"
    # plt.savefig(figure_path)

    # print (figure_path)
    # plt.clf()
