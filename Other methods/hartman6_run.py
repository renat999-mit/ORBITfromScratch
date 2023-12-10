#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  6 19:44:38 2023

@author: renatotronofigueras
"""

import os
import sys
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pickle
from deap import base, creator, tools, algorithms
import random

# Define the Hartman 6 function
def hartman6_function(x):
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([[10, 3, 17, 3.5, 1.7, 8],
                  [0.05, 10, 17, 0.1, 8, 14],
                  [3, 3.5, 1.7, 10, 17, 8],
                  [17, 8, 0.05, 10, 0.1, 14]])
    P = 10 ** (-4) * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                               [2329, 4135, 8307, 3736, 1004, 9991],
                               [2348, 1451, 3522, 2883, 3047, 6650],
                               [4047, 8828, 8732, 5743, 1091, 381]])
    
    result = -np.sum([alpha[j] * np.exp(-np.sum([A[j, i] * (x[i] - P[j, i]) ** 2 for i in range(6)])) for j in range(4)])
    return result,

# Problem size
n_dimensions = 6
bounds = [(0, 1)] * n_dimensions

# Create types
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Initialize functions
toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n_dimensions)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Register evaluation, selection, crossover, and mutation functions
toolbox.register("evaluate", hartman6_function)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.3, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

func_eval = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]

results = np.zeros((30, len(func_eval)))

for i in range(30):
    for j in range(len(func_eval)):
        # Set up GA parameters
        population_size = int(func_eval[j] / 10)
        num_generations = 10
        crossover_prob = 0.8
        mutation_prob = 0.2

        # Initialize population
        pop = toolbox.population(n=population_size)

        # Run GA
        result = algorithms.eaSimple(pop, toolbox, cxpb=crossover_prob, mutpb=mutation_prob, ngen=num_generations, verbose=False)
        
        # Get the best individual
        best_individual = tools.selBest(pop, 1)[0]
        best_fitness = hartman6_function(best_individual)[0]
        
        results[i, j] = best_fitness
        
# compute average values of each column
mean_values = np.mean(results, axis=0)

# compute 0.95 confidence intervals of each column
stderr_values = np.std(results, axis=0, ddof=1) / np.sqrt(30)
conf_intervals = 1.96 * stderr_values

fig, ax = plt.subplots()

ax.set_title('Hartman 6 (d = 6)')
ax.errorbar(func_eval, mean_values, yerr=conf_intervals, fmt='-o', capsize=5, color='k', label='GA')
ax.set_xlabel("Number of function evaluations")
ax.set_ylabel(r"Average of $best~value$ over 30 runs")

result = np.vstack([mean_values, conf_intervals, func_eval])

with open('hartman6_GA_result.pkl', 'wb') as f:
    pickle.dump(result, f)

# Define bounds for the optimization problem
bounds = [(0., 1.), (0., 1.), (0., 1.), (0., 1.), (0., 1.), (0., 1.)]

results = np.zeros(30)

for i in range(30):
    # Generate a random initial guess within the bounds
    x0 = np.array([np.random.uniform(low, high) for low, high in bounds])

    result = sp.optimize.minimize(hartman6_function, x0, method='L-BFGS-B', bounds=bounds)

    results[i] = result.fun

# compute average values of each column
mean_values = np.mean(results)

# compute 0.95 confidence intervals of each column
stderr_values = np.std(results, ddof=1) / np.sqrt(30)
conf_intervals = 1.96 * stderr_values

# ax.axhline(mean_values, color='r', label='L-BFGS-B Optimum Value')
# ax.axhline(mean_values - conf_intervals, color='r', ls='dashed', label='L-BFGS-B Confidence interval')
# ax.axhline(mean_values + conf_intervals, color='r', ls='dashed')
# ax.legend()

# Open the pickle file in read mode
with open('hartman_orbit_result.pkl', 'rb') as f:
    # Load the data from the file using pickle.load
    data_orbit = (pickle.load(f)).T
    
ax.errorbar(data_orbit[:,2],data_orbit[:,0], yerr=data_orbit[:,1], fmt='-o', capsize=5, color = 'b', label = 'ORBIT')
ax.legend()
fig.savefig('hartman6.pdf')