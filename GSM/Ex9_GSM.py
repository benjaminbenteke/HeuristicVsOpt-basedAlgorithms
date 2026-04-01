## Ex9
import numpy as np
from scipy.optimize import minimize

import numpy as np
from scipy.optimize import minimize
from math import *
np.set_printoptions(suppress=True)
## Parallelization
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.optimize import minimize
import joblib
import math
import pandas as pd


nubmer_points_list= [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]
def organize_sol(results):
    solns= []

    for res in results:
        solns.append(res.tolist())

    return solns
import os

for num_points in range(100, 1501, 100):
    path = f"./Ex9/N_{num_points}"
    os.makedirs(path, exist_ok=True)

n_runs= [10]
joblib.cpu_count()

# Define objective functions for each player
def player1_objective(x1, x2):
    return -x1

def player2_objective(x2, x1):
    return 0.0

# Define constraints for each player
def player_constraint11(x1, x2):
    return 1-(x1+x2) # Example: -2 * x1 + x2 - 2 * x3 <= 5

def player_constraint12(x1, x2):
    return 3-(2*x1+4*x2)

def player_constraint21(x2, x1):
    return 1-(x1+x2) # Example: -2 * x1 + x2 - 2 * x3 <= 5

def player_constraint22(x2, x1):
    return 3-(2*x1+4*x2)

# Define the update strategy function for each player
def update_player_strategy(player_index, current_strategies):
    # num_players = len(current_strategies)
    # bound= [(0.0,1.0)]

    if player_index == 0:
        other_players_strategies = np.delete(current_strategies, player_index, axis=0)
        # other_players_strategies = current_strategies[2]
        # print(other_players_strategies)
        objective_function = lambda x: player1_objective(x,other_players_strategies)
        constraint = [{'type': 'ineq', 'fun': lambda x: player_constraint11(x, other_players_strategies)}, {'type': 'ineq', 'fun': lambda x: player_constraint12(x, other_players_strategies)}]
        result = minimize(objective_function, x0= 0.0, constraints=constraint, method= "SLSQP")
        min_x_i = result.x
    else:
        # other_players_strategies = current_strategies[:2]
        other_players_strategies = np.delete(current_strategies, player_index, axis=0)
        # print(other_players_strategies)
        objective_function = lambda x: player2_objective(x,other_players_strategies)
        # print(objective_function(current_strategies[2]))
        constraint = [{'type': 'ineq', 'fun': lambda x: player_constraint21(x, other_players_strategies)}, {'type': 'ineq', 'fun': lambda x: player_constraint22(x, other_players_strategies)}]
        result = minimize(objective_function, x0=0.0, constraints=constraint, bounds= bound,method= "SLSQP") # , constraints=constraint
        min_x_i = result.x

    return min_x_i

def gauss_seidel_gnep(num_players= 2, max_iterations=100, tolerance=1e-6):
    # Initialization
    lower_bnd, upper_bnd= 0.0,1.0 # From shared constraint.
    current_strategies= np.random.uniform(lower_bnd, upper_bnd, size=(num_players,))
    # previous_strategies = np.random.random((num_players,))
    previous_strategies= np.random.random((num_players,))

    # Iterative Update
    iteration = 0
    while iteration < max_iterations and np.linalg.norm(current_strategies - previous_strategies) > tolerance:
        previous_strategies = current_strategies.copy()
        # print(previous_strategies)

        # Update each player's strategy sequentially
        for player_index in range(num_players):
            current_strategies[player_index] = update_player_strategy(player_index, current_strategies)

        iteration += 1

    return current_strategies

# Example with three players and different objective functions and constraints
num_players = 2


num_processes = -1
# num_run= 100
# results = Parallel(n_jobs=num_processes)(delayed(gauss_seidel_gnep)() for _ in range(num_run))
def run_with_diff_start_points(num_points):
    results = Parallel(n_jobs=num_processes)(delayed(gauss_seidel_gnep)() for _ in range(num_points))
    set_of_points= organize_sol(results)
    return set_of_points
n_runs= [10]
def run_with_diff_n_runs(num_points):
    final_res= []
    for n_r in n_runs:
        temp_res= []
        
        for i in range(n_r):
            res= run_with_diff_start_points(num_points)
            temp_res.extend(res)
            res= np.array(res)
            ## Get distinct points
#             num, distinct_points = count_repeated_points(res)
#             distinct_points= np.array(distinct_points)
            np.savetxt('./GSM/solns_runs/Ex10/N_'+str(num_points)+"/"+str(i+1)+"_"+"solns"+'_'+'run_'+str(n_r)+'_'+str(num_points)+'pts'+'.txt', res, delimiter=',')

            
        final_res.append(temp_res)
            
#     results = Parallel(n_jobs=num_processes)(delayed(run_with_diff_start_points)(num_points) for _ in [2,3])
    return final_res

results = Parallel(n_jobs=num_processes)(delayed(run_with_diff_n_runs)(num_points) for num_points in nubmer_points_list)