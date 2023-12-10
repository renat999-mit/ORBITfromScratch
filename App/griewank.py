#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 11:01:24 2023

@author: renatotronofigueras
"""

import os
import sys

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CODE_DIR, ".."))

from Base.orbit_base import OrbitBase

import numpy as np

import math

def griewank10(x):
    """
    Compute the value of the 10-dimensional Griewank function for the given input vector x.
    
    :param x: A list or tuple of 10 numbers representing the input vector x
    :return: The value of the Griewank function for the input vector x
    """
    if len(x) != 10:
        raise ValueError("Input vector must have 10 dimensions")
    
    sum_term = sum(x_i ** 2 for x_i in x) / 4000.0
    product_term = math.prod(math.cos(x_i / math.sqrt(i + 1)) for i, x_i in enumerate(x))
    
    return 1 + sum_term - product_term

class Griewank10(OrbitBase):
    
    def objective_function(self, x):
        
        return griewank10(x)