#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  6 17:30:44 2023

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
from App.rosenbrock_d import rosenbrockd

N_runs = 30

num_func_evaluations = [10,20,30,40,50,60,70,80,90,100,120]

N_points = len(num_func_evaluations) + 1

orbit_problem_name = 'rosenbrockd'

lower_bound = -5.
upper_bound = 5.
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
ax.set_title('Rosenbrock ORBIT (d = 3)')
ax.errorbar(num_func_vec,mean_values, yerr=conf_intervals, fmt='-o', capsize=5, color = 'k', label = 'My implementation')
ax.set_xlabel("Number of function evaluations")
ax.set_ylabel(r"Average of $best~value$ over 30 runs")
ax.legend()
# fig.savefig('hartman3.pdf')

result = np.vstack([mean_values,conf_intervals,num_func_vec])

with open('rosenbrock_orbit_result.pkl', 'wb') as f:
    pickle.dump(result, f)