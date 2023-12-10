#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 16:02:28 2023

@author: renatotronofigueras
"""

import os
import sys
import inspect

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CODE_DIR, ".."))

import numpy as np

from Base.orbit_base import OrbitBase
from Util.initialize_case import initialize_case
# from App import goldstein_price
# from App import hartman3
# from App import shekel5
from App import *


def run_cycle(N_runs,N_points,num_func_evaluations,orbit_problem_name,base_par):
    
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        
        if inspect.ismodule(obj):
            
            try:
                problem_name = getattr(obj, orbit_problem_name)
                
                break
            
            except AttributeError:
                
                pass
    
    pre_result = np.zeros((N_runs,N_points))
    
    num_evaluations_vector = np.zeros(N_points)
    
    initial_parameters = initialize_case(*base_par)
    
    for run_i in range(N_runs):
        
        print("Run: ", run_i + 1)
        
        test_case = problem_name(*initial_parameters)
        
        pre_result[run_i,0] = test_case.get_current_solution()[1]
        
        print(test_case.get_current_solution()[1])
        
        if run_i == 0:
        
            num_evaluations_vector[0] = test_case.n_
        
        counter_spacing = 0
        
        counter_points = 1
        
        while(test_case.n_ <= test_case.max_evaluations_):
            
            test_case.single_step()
            
            if test_case.n_ in num_func_evaluations:
                
                pre_result[run_i,counter_points] = test_case.get_current_solution()[1]
                
                print(test_case.get_current_solution()[1])
                
                if run_i == 0:
                
                    num_evaluations_vector[counter_points] = test_case.n_
                
                counter_points += 1
                
    if orbit_problem_name == 'GoldsteinPrice':
        
        pre_result = np.log(pre_result)
                
    # compute average values of each column
    means_values = np.mean(pre_result, axis=0)
    
    # compute 0.95 confidence intervals of each column
    stderr_values = np.std(pre_result, axis=0, ddof=1) / np.sqrt(N_runs)
    conf_intervals = 1.96 * stderr_values
    
    return means_values, conf_intervals, num_evaluations_vector
        