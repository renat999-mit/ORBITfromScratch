#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  6 13:44:21 2023

@author: renatotronofigueras
"""
import os
import sys

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CODE_DIR, ".."))

from Base.orbit_base import OrbitBase

import numpy as np

def rastrigin_function(x, A=10):
    n = len(x)
    
    return A * n + np.sum([xi**2 - A * np.cos(2 * np.pi * xi) for xi in x])

class rastring3d(OrbitBase):
    
    def objective_function(self, x):
        
        return rastrigin_function(x)
    
