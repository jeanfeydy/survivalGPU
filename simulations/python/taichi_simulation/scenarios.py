import numpy as np 

def exponential_scenario(u_t, name = False):
    return((7 * np.exp(-7*u_t/365)*0.5)) # divide by 365 in order to have a t in days

def test_opposite_square_scenario(u_t):
    return 1/np.exp(u_t)


def get_scenario(scenario_name: int,cutoff:int):
    """
    For each scenario function implemeted, this function will take into input the scenario name and the cutoff
    and return the list of the scenario shape normalized so that the sum of the weights is 
    equal to 1.

    The scenario function mus be defined and added to the dicitonanry scenario list
    """

    scenario_list = {
        "exponential_scenario" : exponential_scenario,
        "test_opposite_square_scenario" : test_opposite_square_scenario
    }

    try:
        scenario_function = scenario_list[scenario_name]
    except KeyError:
        print(f"The scenario {scenario_name} is not defined")


    scenario_list = []

    for i in range(1,cutoff+1):
        scenario_list.append(scenario_function(i)) 
    scenario_list = np.array(scenario_list)


    return scenario_list/scenario_list.sum()

        