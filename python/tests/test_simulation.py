import torch
import numpy as np
from survivalgpu.simulation import get_probas
from survivalgpu.simulation import event_FUP_Ti_generation
from survivalgpu.simulation import generate_wce_mat
from survivalgpu.simulation import matching_algo
from survivalgpu.simulation import get_dataset
from survivalgpu.simulation import get_scenario
from scipy.stats import norm

import pytest
import pandas as pd


def test_get_scenario():

    # For exponential scenario


    expected_scenario_shape = np.zeros(365)
    for i in range(365):
        expected_scenario_shape[i] = 7*np.exp(-7*i/365)
    
    normalization_factor = np.sum(expected_scenario_shape)
    expected_scenario_shape = expected_scenario_shape/normalization_factor
    real_scenario_shape = get_scenario("exponential_scenario", 365)
    assert np.allclose(real_scenario_shape, expected_scenario_shape)



    # For bi-linear scenario


    expected_scenario_shape = np.zeros(365)
    for i in range(365):
        if i < 50:
            expected_scenario_shape[i] = 1- (i/365)/(50/365)
        else:
            expected_scenario_shape[i] = 0

    normalization_factor = np.sum(expected_scenario_shape)
    expected_scenario_shape = expected_scenario_shape/normalization_factor
    real_scenario_shape = get_scenario("bi_linear_scenario", 365)
    assert np.allclose(real_scenario_shape, expected_scenario_shape)

    # For early peak scenario

    expected_scenario_shape = np.zeros(365)
    for i in range(365):
        expected_scenario_shape[i] = norm.pdf(i/365, 0.04, 0.05)

    
    normalization_factor = np.sum(expected_scenario_shape)
    expected_scenario_shape = expected_scenario_shape/normalization_factor
    real_scenario_shape = get_scenario("early_peak_scenario", 365)
    assert np.allclose(real_scenario_shape, expected_scenario_shape)

    # for inverted u scenario

    expected_scenario_shape = np.zeros(365)
    for i in range(365):
        expected_scenario_shape[i] = norm.pdf(i/365, 0.2, 0.06)

    normalization_factor = np.sum(expected_scenario_shape)
    expected_scenario_shape = expected_scenario_shape/normalization_factor
    real_scenario_shape = get_scenario("inverted_u_scenario", 365)
    assert np.allclose(real_scenario_shape, expected_scenario_shape)


    # for a function that is not implemented

def test_get_scenario_not_implemented():
    # For non-existent scenario
    with pytest.raises(ValueError, match="The scenario 'unknown_scenario' is not defined"):
        get_scenario("unknown_scenario", 365)


def test_generate_wce_mat():

    scenario = "exponential_scenario"

    Xmat = np.array([[1, 2, 0, 5], [4, 0, 0, 6], [7, 8, 1, 0]])

    max_time = Xmat.shape[0]
    n_patient = Xmat.shape[1]
    
    print(n_patient)

    exponential_shape = 7*np.exp(-7*np.arange(0,365,1)/365)
    normalization_factor = np.sum(exponential_shape)
    print(normalization_factor)

    print(exponential_shape)
    normalization_factor = np.sum(exponential_shape)
    exponential_shape = exponential_shape/normalization_factor

    # expected_wce_mat = Xmat * exponential_shape[:n_patient].transpose()

    wce_mat = generate_wce_mat(scenario,Xmat, max_time)

    wce_mat_expected = np.zeros((max_time, n_patient))

    for i in range(1,max_time +1):
        for j in range(n_patient):

            wce_factor = 0

            for time in range(1, i +1):
                wce_factor += Xmat[time - 1,j] * exponential_shape[i-time]

            wce_mat_expected[i-1,j] = wce_factor

    assert np.allclose(wce_mat, wce_mat_expected)




def test_event_FUP_Ti_generation():
    eventRandom = np.array([3, 5, 10])
    censorRandom = np.array([5, 2, 10])

    exepected_event = np.array([0, 1, 1])
    expected_FUP_Ti = np.array([2, 3, 10])
    

    event, FUP_Ti = event_FUP_Ti_generation(eventRandom, censorRandom)

    assert np.allclose(event, exepected_event)
    assert np.allclose(FUP_Ti, expected_FUP_Ti)    




def test_get_probas():


    def manual_proba_calc(vector, HR_target):

        return_vector = []

        for patient in vector:
            prba = np.exp(np.log(HR_target) * patient) / sum(np.exp(np.log(HR_target) * vector))
            return_vector.append(prba)
        
        return torch.tensor(return_vector)
    

    # Test case 1 : time = 1
    wce_mat_current = torch.rand(5,6)
    HR_target = 1.5
    time_event = 1

    expected_probas = manual_proba_calc(wce_mat_current[time_event -1], HR_target)
    probas = get_probas(wce_mat_current, HR_target, time_event)
    # Test case 2 : time = 2
    wce_mat_current = torch.rand(5,6)
    HR_target = 1.5
    time_event = 2

    expected_probas = manual_proba_calc(wce_mat_current[time_event -1], HR_target)
    probas = get_probas(wce_mat_current, HR_target, time_event)       
    assert torch.allclose(probas, expected_probas)


    # Test case 3 : HR target = 1

    wce_mat_current = torch.rand(5,6)
    HR_target = 1
    time_event = 1

    # If time = 3 the expected probabilities should be 1/3 for each patient, whatever is the WCE matrix and the time of event
    expected_probas = torch.ones(5)/5

    probas = get_probas(wce_mat_current, HR_target, time_event)
    # Test case 4 : empty matrix

    wce_mat_current = torch.zeros(5,6)
    HR_target = 1
    time_event = 1

    # If time = 3 the expected probabilities should be 1/3 for each patient, whatever is the WCE matrix and the time of event
    expected_probas = torch.tensor([1/5, 1/5, 1/5, 1/5, 1/5]) 

    probas = get_probas(wce_mat_current, HR_target, time_event)

def test_matching_algo():

    wce_mat = np.random.rand(20,30)
    max_time = 20
    n_patients = 30
    HR_target = 1.5
    event = np.random.randint(2, size=n_patients)
    FUP_tis = np.sort(np.random.randint(1, max_time, size=n_patients))


    wce_id_indexes = matching_algo(wce_mat, max_time, n_patients, HR_target, event, FUP_tis)

    ordered_wce_id_indexes = np.sort(wce_id_indexes)

    ordered_expected_id_indexes = np.arange(0, n_patients, 1)
    print("OK")

    assert np.allclose(ordered_wce_id_indexes, ordered_expected_id_indexes)




def test_get_dataset():

    n_patients = 4
    max_time = 5
    HR_target = 1.5


    Xmat = np.array([[1, 2, 0, 0],
                     [2, 3, 1, 1],
                     [0, 2, 2, 2],
                     [0, 5, 0, 3],
                     [0, 5, 0, 3]])
    print(Xmat)

    FUP_tis = [3,2,1,2]
    events = [1,0,0,1]

    wce_id_indexes = [2,0,3,1]


    expected_result_numpy = np.array([[1,0,1,0,0],
                                      [1,1,2,0,1],
                                      [1,2,3,1,2],
                                      [2,0,1,0,1],
                                      [2,1,2,0,2],
                                      [3,0,1,0,0],
                                      [4,0,1,0,2],
                                      [4,1,2,1,3],
                                      ])
    
    expected_result_dataframe = pd.DataFrame(expected_result_numpy, columns=["patient","start","stop","event","dose"])
    
    print("OK")
    print(expected_result_dataframe)

    real_result_dataframe = get_dataset(Xmat, max_time, n_patients, HR_target, FUP_tis, events, wce_id_indexes)

    print(real_result_dataframe)

    assert np.allclose(expected_result_dataframe, real_result_dataframe)

    




    
test_get_scenario()
test_get_scenario_not_implemented()
test_generate_wce_mat()
test_event_FUP_Ti_generation()
test_get_probas()
test_matching_algo()
test_get_dataset()





