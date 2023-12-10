#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 09:57:27 2023

@author: renatotronofigueras
"""

import os
import sys

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CODE_DIR, ".."))

from Base.orbit_base import OrbitBase

import numpy as np

def shekel5(x):
    A = np.array([
        [4, 4, 4, 4],
        [1, 1, 1, 1],
        [8, 8, 8, 8],
        [6, 6, 6, 6],
        [3, 7, 3, 7]
    ])

    c = np.array([0.1, 0.2, 0.2, 0.4, 0.4])
    m = 5

    return -np.sum([1 / (np.sum((x - A[j, :])**2) + c[j]) for j in range(m)])

class Shekel5(OrbitBase):
    def objective_function(self, x):
        return shekel5(x)

