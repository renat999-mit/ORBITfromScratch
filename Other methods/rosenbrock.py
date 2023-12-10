#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  6 16:38:34 2023

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

# Define the Rosenbrock function
def rosenbrock_function(individual):
    return sum(100 * (individual[i + 1] - individual[i] ** 2) ** 2 + (1 - individual[i]) ** 2 for i in range(len(individual) - 1)),

# Define the gradient of the Rosenbrock function
def rosenbrock_gradient(x):
    n = len(x)
    grad = np.zeros(n)
    for i in range(n - 1):
        grad[i] = -400 * (x[i + 1] - x[i] ** 2) * x[i] + 2 * (x[i] - 1)
        grad[i + 1] += 200 * (x[i + 1] - x[i] ** 2)
    return grad

# Problem size
n_dimensions = 3
bounds = [(-5, 5)] * n_dimensions

# Create types
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Initialize functions
toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -5.12, 5.12)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n_dimensions)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Register evaluation, selection, crossover, and mutation functions
toolbox.register("evaluate", rosenbrock_function)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.3, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

func_eval = [10,20,30,40,50,60,70,80,90,100,120]

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
        best_fitness = rosenbrock_function(best_individual)[0]
        
        results[i,j] = best_fitness
        
# compute average values of each column
mean_values1 = np.mean(results, axis=0)

# compute 0.95 confidence intervals of each column
stderr_values1 = np.std(results, axis=0, ddof=1) / np.sqrt(30)
conf_intervals1 = 1.96 * stderr_values1


result = np.vstack([mean_values1,conf_intervals1,func_eval])

with open('rosenbrock_GA_result.pkl', 'wb') as f:
    pickle.dump(result, f)

# Define bounds for the optimization problem
bounds = [(-5., 5.), (-5., 5.), (-5., 5.)]

results = np.zeros(30)

for i in range(30):
    
    
    # Generate a random initial guess within the bounds
    x0 = np.array([np.random.uniform(low, high) for low, high in bounds])

    result = sp.optimize.minimize(rosenbrock_function, x0, method='L-BFGS-B', bounds=bounds)

    results[i] = result.fun
    

# compute average values of each column
mean_values = np.mean(results)

# compute 0.95 confidence intervals of each column
stderr_values = np.std(results, ddof=1) / np.sqrt(30)
conf_intervals = 1.96 * stderr_values


# Open the pickle file in read mode
with open('rosenbrock_orbit_result.pkl', 'rb') as f:
    # Load the data from the file using pickle.load
    data_orbit = (pickle.load(f)).T

fig, ax = plt.subplots()

ax.set_title('Rosenbrock (d = 3)')
ax.errorbar(func_eval,mean_values1, yerr=conf_intervals1, fmt='-o', capsize=5, color = 'k', label = 'GA')
ax.set_xlabel("Number of function evaluations")
ax.set_ylabel(r"Average of $best~value$ over 30 runs")
# ax.set_ylim(0,100)
ax.axhline(mean_values, color = 'r', label = 'L-BFGS-B Optimum Value')
ax.axhline(mean_values - conf_intervals, color = 'r', ls = 'dashed', label = 'L-BFGS-B Confidence interval')
ax.axhline(mean_values + conf_intervals, color = 'r', ls = 'dashed')
ax.legend()
ax.errorbar(data_orbit[:,2],data_orbit[:,0], yerr=data_orbit[:,1], fmt='-o', capsize=5, color = 'b', label = 'ORBIT')
ax.legend()
fig.savefig('rosenbrock.pdf')

fig1, ax1 = plt.subplots()

ax1.set_title('Rosenbrock (d = 3)')
ax1.errorbar(func_eval,mean_values1, yerr=conf_intervals1, fmt='-o', capsize=5, color = 'k', label = 'GA')
ax1.set_xlabel("Number of function evaluations")
ax1.set_ylabel(r"Average of $best~value$ over 30 runs")
# ax.set_ylim(0,100)
ax1.axhline(mean_values, color = 'r', label = 'L-BFGS-B Optimum Value')
ax1.axhline(mean_values - conf_intervals, color = 'r', ls = 'dashed', label = 'L-BFGS-B Confidence interval')
ax1.axhline(mean_values + conf_intervals, color = 'r', ls = 'dashed')
ax1.set_ylim(-1,100)
ax1.errorbar(data_orbit[:,2],data_orbit[:,0], yerr=data_orbit[:,1], fmt='-o', capsize=5, color = 'b', label = 'ORBIT')
ax1.legend()
fig1.savefig('rosenbrock_zoom.pdf')
