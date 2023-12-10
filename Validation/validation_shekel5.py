#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 09:58:47 2023

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
from App.shekel5 import Shekel5

N_runs = 30
N_points = 11

num_func_evaluations = [20,40,60,80,100,120,140,160,180,200]

orbit_problem_name = 'Shekel5'

lower_bound = 0
upper_bound = 10
kernel_func_type = 'surface spline'
d = 4
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

#%%

plt.rcParams['text.usetex'] = True

import pickle

paper_data = np.loadtxt('shekel5_paper.txt')
# # Open the pickle file in read mode
# with open('shekel5_result.pkl', 'rb') as f:
#     # Load the data from the file using pickle.load
#     data_orbit = (pickle.load(f)).T

fig, ax = plt.subplots()
ax.set_title('Shekel5 ($d = 4$)')
ax.errorbar(num_func_vec,mean_values, yerr=conf_intervals, fmt='-o', capsize=5, color = 'k', label = 'My implementation')
ax.plot(paper_data[:,0],paper_data[:,1], '-o',label = 'Regis et al. 2007')
ax.set_xlabel("Number of function evaluations")
ax.set_ylabel(r"Average of best value over 30 runs")
ax.legend()
fig.savefig('shekel5.pdf')

result = np.vstack([mean_values,conf_intervals,num_func_vec])

with open('shekel5_result.pkl', 'wb') as f:
    pickle.dump(result, f)