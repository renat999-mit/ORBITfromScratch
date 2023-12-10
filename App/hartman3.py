#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 09:32:13 2023

@author: renatotronofigueras
"""

import os
import sys

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CODE_DIR, ".."))

from Base.orbit_base import OrbitBase

import numpy as np

class Hartman3(OrbitBase):

    def objective_function(self, x):
        alpha = np.array([1.0, 1.2, 3.0, 3.2])
        A = np.array([[3.0, 10.0, 30.0],
                      [0.1, 10.0, 35.0],
                      [3.0, 10.0, 30.0],
                      [0.1, 10.0, 35.0]])
        P = 0.0001 * np.array([[3689, 1170, 2673],
                                [4699, 4387, 7470],
                                [1091, 8732, 5547],
                                [381, 5743, 8828]])

        outer_sum = 0
        for i in range(4):
            inner_sum = 0
            for j in range(3):
                inner_sum += A[i, j] * (x[j] - P[i, j])**2
            outer_sum += alpha[i] * np.exp(-inner_sum)
        return -outer_sum