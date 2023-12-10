#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 16:01:46 2023

@author: renatotronofigueras
"""

import os
import sys
import numpy as np

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CODE_DIR, ".."))

def initialize_case(
        lower_bound,
        upper_bound,
        kernel_func_type,
        d,
        gamma = None,
        kappa = None,
        n0 = None,
        t = None,
        w_r_pattern = None,
        max_evaluations = 100,
        poly = False
        ):
    
    lower_bounds = np.full(d, lower_bound)
    upper_bounds = np.full(d, upper_bound) 
    
    if t is None:
        
        t = 1000*d
        
    if n0 is None:
        
        n0 = 2*(d+1)
        
    An0 = np.random.uniform(lower_bounds, upper_bounds, size=(n0, d))
    
    kernel_type = {'type': kernel_func_type}
        
    if gamma is not None:
        
        kernel_type['gamma'] = gamma
        
    if kappa is not None:
        
        kernel_type['kappa'] = kappa
        
    if w_r_pattern is None:
        
        w_r_pattern = np.array([0.2, 0.4, 0.6, 0.9, 0.95, 1])
        
    return [lower_bounds,upper_bounds,An0,max_evaluations,t,kernel_type,w_r_pattern,poly]