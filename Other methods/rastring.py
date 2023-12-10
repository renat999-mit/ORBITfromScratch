#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  6 14:01:19 2023

@author: renatotronofigueras
"""

import os
import sys

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CODE_DIR, ".."))

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pickle
from deap import base, creator, tools, algorithms
import random

# Define the Rastrigin function
def rastrigin_function(individual, A=10):
    n = len(individual)
    return A * n + sum([x**2 - A * np.cos(2 * np.pi * x) for x in individual]),

# Problem size
n_dimensions = 3
bounds = [(-5.12, 5.12)] * n_dimensions

# Create types
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Initialize functions
toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -5.12, 5.12)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n_dimensions)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Register evaluation, selection, crossover, and mutation functions
toolbox.register("evaluate", rastrigin_function)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.3, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)






func_eval = [20,40,60,80,100,120,140,160,180,200]

results = np.zeros((30,len(func_eval)))

for i in range(30):
    
    for j in range(len(func_eval)):
        
        # Set up GA parameters
        population_size = int(func_eval[j]/10)
        num_generations = 10
        crossover_prob = 0.8
        mutation_prob = 0.2
        
        # Initialize population
        pop = toolbox.population(n=population_size)
        
        # Run GA
        result = algorithms.eaSimple(pop, toolbox, cxpb=crossover_prob, mutpb=mutation_prob, ngen=num_generations, verbose=False)
        
        # Get the best individual
        best_individual = tools.selBest(pop, 1)[0]
        best_fitness = rastrigin_function(best_individual)[0]
        
        results[i,j] = best_fitness
        
# compute average values of each column
mean_values = np.mean(results, axis=0)

# compute 0.95 confidence intervals of each column
stderr_values = np.std(results, axis=0, ddof=1) / np.sqrt(30)
conf_intervals = 1.96 * stderr_values

fig, ax = plt.subplots()

ax.set_title('Rastrigin (d = 3)')
ax.errorbar(func_eval,mean_values, yerr=conf_intervals, fmt='-o', capsize=5, color = 'k', label = 'GA')
ax.set_xlabel("Number of function evaluations")
ax.set_ylabel(r"Average of $best~value$ over 30 runs")
# fig.savefig('hartman3.pdf')

result = np.vstack([mean_values,conf_intervals,func_eval])

with open('rastring_GA_result.pkl', 'wb') as f:
    pickle.dump(result, f)
    

# Define the gradient of the Rastrigin function
def rastrigin_gradient(x, A=10):
    n = len(x)
    return np.array([2 * x[i] - 2 * A * np.pi * np.sin(2 * np.pi * x[i]) for i in range(n)])

# Define bounds for the optimization problem
bounds = [(-5.12, 5.12), (-5.12, 5.12), (-5.12, 5.12)]



results = np.zeros(30)

for i in range(30):
    
    max_iterations = 200
    
    # Generate a random initial guess within the bounds
    x0 = np.array([np.random.uniform(low, high) for low, high in bounds])

    result = sp.optimize.minimize(rastrigin_function, x0, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': max_iterations})

    results[i] = result.fun
    

# compute average values of each column
mean_values = np.mean(results)

# compute 0.95 confidence intervals of each column
stderr_values = np.std(results, ddof=1) / np.sqrt(30)
conf_intervals = 1.96 * stderr_values

ax.axhline(mean_values, color = 'r', label = 'L-BFGS-B Optimum Value')
ax.axhline(mean_values - conf_intervals, color = 'r', ls = 'dashed', label = 'L-BFGS-B Confidence interval')
ax.axhline(mean_values + conf_intervals, color = 'r', ls = 'dashed')
# ax.legend()


# Open the pickle file in read mode
with open('rastrigin_orbit_result.pkl', 'rb') as f:
    # Load the data from the file using pickle.load
    data_orbit = (pickle.load(f)).T
    
ax.errorbar(data_orbit[:,2],data_orbit[:,0], yerr=data_orbit[:,1], fmt='-o', capsize=5, color = 'b', label = 'ORBIT')
ax.legend()
fig.savefig('rastrigin3.pdf')



        



