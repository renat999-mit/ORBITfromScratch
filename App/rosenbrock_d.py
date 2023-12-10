#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  6 16:45:39 2023

@author: renatotronofigueras
"""

import os
import sys

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CODE_DIR, ".."))

from Base.orbit_base import OrbitBase

import numpy as np

def rosenbrock_function(x):
    return sum(100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2 for i in range(len(x) - 1))


class rosenbrockd(OrbitBase):
    
    def objective_function(self, x):
        
        return rosenbrock_function(x)