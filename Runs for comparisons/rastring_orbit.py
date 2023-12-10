#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  6 13:49:01 2023

@author: renatotronofigueras
"""

import os
import sys

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CODE_DIR, ".."))

import numpy as np
import matplotlib.pyplot as plt

from Base.orbit_base import OrbitBase
from Util.initialize_case import initialize_case
from Util.run_cycle import run_cycle
from App.rastring3 import rastring3d

N_runs = 30
N_points = 11

num_func_evaluations = [20,40,60,80,100,120,140,160,180,200]

orbit_problem_name = 'rastring3d'

lower_bound = -5.12
upper_bound = 5.12
kernel_func_type = 'surface spline'
d = 3
gamma = None
kappa = 2
n0 = None
t = None
w_r_pattern = None
max_evaluations = 200
poly = True

base_par = [lower_bound,
            upper_bound,
            kernel_func_type,
            d,
            gamma,
            kappa,
            n0,
            t,
            w_r_pattern,
            max_evaluations,
            poly]

mean_values, conf_intervals, num_func_vec = run_cycle(N_runs, N_points, num_func_evaluations, orbit_problem_name, base_par)

import pickle

fig, ax = plt.subplots()
ax.set_title('Rastring ORBIT (d = 3)')
ax.errorbar(num_func_vec,mean_values, yerr=conf_intervals, fmt='-o', capsize=5, color = 'k', label = 'My implementation')
ax.set_xlabel("Number of function evaluations")
ax.set_ylabel(r"Average of $best~value$ over 30 runs")
ax.legend()
# fig.savefig('hartman3.pdf')

result = np.vstack([mean_values,conf_intervals,num_func_vec])

with open('rastring_orbit_result.pkl', 'wb') as f:
    pickle.dump(result, f)