#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 10:59:31 2023

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
from App.griewank import Griewank10

N_runs = 30
N_points = 10

num_func_evaluations = [15,20,30,40,50,60,70,80,90,100]

orbit_problem_name = 'Griewank10'

lower_bound = -500
upper_bound = 700
kernel_func_type = 'surface spline'
d = 10
gamma = None
kappa = 2
n0 = None
t = None
w_r_pattern = None
max_evaluations = 100
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

paper_data = np.loadtxt('griewank10_paper.txt')
# # Open the pickle file in read mode
# with open('goldstein_price_result.pkl', 'rb') as f:
#     # Load the data from the file using pickle.load
#     data_orbit = (pickle.load(f)).T

fig, ax = plt.subplots()
ax.set_title('Griewank10 ($d = 10$)')
ax.errorbar(num_func_vec[:-1],mean_values[:-1], yerr=conf_intervals[:-1], fmt='-o', capsize=5, color = 'k', label = 'My implementation')
ax.plot(paper_data[:,0],paper_data[:,1], '-o',label = 'Regis et al. 2007')
ax.set_xlabel("Number of function evaluations")
ax.set_ylabel(r"Average of best value over 30 runs")
ax.legend()
fig.savefig('griewank10.pdf')

result = np.vstack([mean_values,conf_intervals,num_func_vec])

with open('goldstein_price_result.pkl', 'wb') as f:
    pickle.dump(result, f)