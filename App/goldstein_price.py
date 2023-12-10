#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 16:56:35 2023

@author: renatotronofigueras
"""

import os
import sys

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CODE_DIR, ".."))

from Base.orbit_base import OrbitBase

class GoldsteinPrice(OrbitBase):

    def objective_function(self, x):
        x1 = x[0]
        x2 = x[1]
        
        term1 = (1 + (x1 + x2 + 1)**2 * (19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2))
        term2 = (30 + (2*x1 - 3*x2)**2 * (18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2))
        
        return term1 * term2