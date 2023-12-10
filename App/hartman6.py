#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  6 19:39:08 2023

@author: renatotronofigueras
"""

import os
import sys

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CODE_DIR, ".."))

from Base.orbit_base import OrbitBase

import numpy as np

def hartman6_function(x):
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([[10, 3, 17, 3.5, 1.7, 8],
                  [0.05, 10, 17, 0.1, 8, 14],
                  [3, 3.5, 1.7, 10, 17, 8],
                  [17, 8, 0.05, 10, 0.1, 14]])
    P = 10 ** (-4) * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                               [2329, 4135, 8307, 3736, 1004, 9991],
                               [2348, 1451, 3522, 2883, 3047, 6650],
                               [4047, 8828, 8732, 5743, 1091, 381]])
    
    result = -np.sum([alpha[j] * np.exp(-np.sum([A[j, i] * (x[i] - P[j, i]) ** 2 for i in range(6)])) for j in range(4)])
    return result

class hartman6(OrbitBase):
    
    def objective_function(self, x):
        
        return hartman6_function(x)