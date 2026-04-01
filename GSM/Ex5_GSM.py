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
    path = f"./Ex5/N_{num_points}"
    os.makedirs(path, exist_ok=True)

n_runs= [10]
joblib.cpu_count()

import numpy as np
from scipy.optimize import minimize

# Define objective functions for each player
import numpy as np
from scipy.optimize import minimize

# Define objective functions for each player
def player1_objective(x1, x2):
    return x1**2-x1*x2-x1

def player2_objective(x2, x1):
    return x2**2-0.5*x1*x2-2*x2

# Define constraints for each player
import numpy as np
from scipy.optimize import minimize

# Define objective functions for each player
import numpy as np
from scipy.optimize import minimize

# Define the objective functions and constraints for each player
def player1_objective(x,x3):
    return x[0]**2+x[0]*x[1]+x[1]**2+(x[0]+x[1])*x3-25*x[0]-38*x[1]

def player2_objective(x3,x):
    return x3**2+(x[0]+x[1])*x[1]-25*x3

def player_constraint11(x,x3):
    # Define the constraint for player i (example: linear constraint)
    return 14-(x[0]+2*x[1]-x3)
def player_constraint12(x, x3):
    # Define the constraint for player i (example: linear constraint)
    return 30-(3*x[0]+2*x[1]+x3)

def player_constraint21(x3,x):
    # Define the constraint for player i (example: linear constraint)
    return 14-(x[0]+2*x[1]-x3)
def player_constraint22(x3,x):
    # Define the constraint for player i (example: linear constraint)
    return 30-(3*x[0]+2*x[1]+x3)


# Gauss-Seidel-type update rule for each player
def update_player_strategy(player_index, current_strategies):
    num_players = len(current_strategies)

    if player_index == 0:
        # other_players_strategies = np.delete(current_strategies, player_index, axis=0)
        other_players_strategies = current_strategies[2]
        # print(other_players_strategies)
        objective_function = lambda x: player1_objective(x,other_players_strategies)
        constraint = [{'type': 'ineq', 'fun': lambda x: player_constraint11(x, other_players_strategies)},{'type': 'ineq', 'fun': lambda x: player_constraint12(x, other_players_strategies)}]
        result = minimize(objective_function, x0= (0.0,0.0), constraints=constraint, bounds= ((0.0,8.0), (3.0,11.0)))
        min_x_i = result.x
    else:
        other_players_strategies = current_strategies[:2]
        # print(other_players_strategies)
        objective_function = lambda x: player2_objective(x,other_players_strategies)
        # print(objective_function(current_strategies[2]))
        constraint = [{'type': 'ineq', 'fun': lambda x: player_constraint21(x,other_players_strategies)},{'type': 'ineq', 'fun': lambda x: player_constraint22(x,other_players_strategies)}]
        result = minimize(objective_function, x0=0.0, constraints=constraint, bounds= [(0.0,8.0)]) # , constraints=constraint
        min_x_i = result.x[0]

    return min_x_i

# Gauss-Seidel-type method for GNEPs
def gauss_seidel_gnep(max_iterations=100, tolerance=1e-6):
    # Initialization
    # current_strategies = np.zeros(3)
    # previous_strategies = np.ones(3)

    x = np.random.uniform(0, 8)
    y = np.random.uniform(3, 11)
    z = np.random.uniform(0, 8)
    z0 = np.array([x, y, z])
    current_strategies = np.array(z0)

    # current_strategies = np.random.random((3,))
    previous_strategies = np.random.random((3,))


    # Iterative Update
    iteration = 0
    # for _ in range(10):
    while iteration < max_iterations and np.linalg.norm(current_strategies - previous_strategies) > tolerance:
        previous_strategies = current_strategies.copy()

        # Update each player's strategy sequentially
        for player_index in range(2):
          if player_index==0:
            current_strategies[:2]= update_player_strategy(player_index, current_strategies)
          else:
            current_strategies[2]= update_player_strategy(player_index, current_strategies)
        iteration += 1

    return current_strategies

num_processes = -1

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
            np.savetxt('./Ex5/N_'+str(num_points)+"/"+str(i+1)+"_"+"solns"+'_'+'run_'+str(n_r)+'_'+str(num_points)+'pts'+'.txt', res, delimiter=',')

            
        final_res.append(temp_res)
            
    return final_res

results = Parallel(n_jobs=num_processes)(delayed(run_with_diff_n_runs)(num_points) for num_points in nubmer_points_list)