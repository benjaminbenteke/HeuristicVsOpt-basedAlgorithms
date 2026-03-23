
import os

percent_list = list(range(10,101,10))

nubmer_points_list = [
    100,200,300,400,500,600,700,800,
    900,1000,1100,1200,1300,1400,1500
]

base_folder = "percent"

for p in percent_list:
    
    for num_points in nubmer_points_list:
        
        path = f"{base_folder}/{p}/Ex3/N_{num_points}"
        
        os.makedirs(path, exist_ok=True)

print("All folders created successfully.")

import numpy as np
from scipy.optimize import minimize
import pandas as pd
import itertools
import numpy as np
import random
import math
from math import *

import time
import sys
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint
from scipy import stats
from sklearn.linear_model import LinearRegression
from numpy import arange
from matplotlib import pyplot
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize_scalar
from sklearn import linear_model

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import ks_2samp
from ctypes import set_errno

import joblib
from joblib import Parallel, delayed
joblib.cpu_count()
num_processes = -1
# np.set_printoptions(precision=1064)
np.set_printoptions(suppress=True)

from itertools import combinations
import joblib
from joblib import Parallel, delayed
joblib.cpu_count()
num_processes = -1

eps_t= 1e-5
delta= 1e-2
def exitCon(pen):
    return np.all(pen<delta)

def calculate_distance(point1, point2):
    # Calculate the Euclidean distance between two points of any dimension
    return sum((x - y) ** 2 for x, y in zip(point1, point2)) ** 0.5
delta= 1e-2
def count_repeated_points(points):
    # Initialize a list to store distinct points
    distinct_points = []

    # Iterate through each point in the set
    for point in points:
        # Check if the point is distinct from all previously considered distinct points
        is_distinct = True
        for distinct_point in distinct_points:
            if calculate_distance(point, distinct_point) < delta:
                is_distinct = False
                break
        # If the point is distinct, add it to the list of distinct points
        if is_distinct:
            distinct_points.append(list(point))
            
    return len(distinct_points), np.array(distinct_points)


## Ex3
def pen(a,b):

  def func1(x):
    return (x)**2 - x*b - x

  def func2(x):
    return (x)**2 - 0.5*a*x - 2*x

  def g1(x):
    return 1-x**2-b**2

  def g2(x):
    return 1-x**2-a**2

  bound1 = [(0.0,math.sqrt(abs(1.0-(b)**2)))]
  bound2 = [(0.0,math.sqrt(abs(1.0-(a)**2)))]

  constraint1 = {'type': 'ineq', 'fun': g1}
  constraint2 = {'type': 'ineq', 'fun': g2}

  result1 = minimize(func1, x0=0.0, method='SLSQP', bounds = bound1, constraints= constraint1) #, options={'maxiter':5}
  result2 = minimize(func2, x0=0.0, method='SLSQP', bounds = bound2,constraints= constraint2)

  shadowX = result1.x
  shadowY = result2.x

  penalty = math.sqrt((math.pow(a-shadowX,2))+(math.pow(b-shadowY,2)))
  return penalty

num_points= 200 #len(grid_points)
n_generations= 1000
perc= 25

def run_example(perc, num_points):
    num = 1
    def minSearch(xMin):
        return xMin - (xMin - 0)*(math.pow(0.99,num))
    def maxSearch(xMax):
        return xMax + (1 - xMax)*(math.pow(0.99,num))
    def maxYSearch(yMax):
        return yMax + (1 - yMax)*(math.pow(0.99,num))
    
    transforming = PolynomialFeatures(degree=2, include_bias = False)

    def poly(x):
        return coef1*math.pow(x,2) + coef2*x + yInt

    def randomY(maxY, minY):
        return random.uniform(minY, maxY)
    # initializing all of the points
    
    # n_delete= int((25*num_points)/100) #30 # number of points to select
    n_delete= int((perc*num_points)/100) #30 # number of points to select

    setXYP = np.zeros(3*num_points)
    setXYP = setXYP.reshape(num_points,3) # 2 for n and 1 is the fitness value of points.

    setX = np.empty(num_points, dtype=object)
    setY = np.empty(num_points, dtype=object)
    penPts = np.zeros(num_points, dtype=object)
    ptNumbers = np.zeros(num_points, dtype=object)
    penSelection = np.empty(n_delete, dtype=object)
    penNo = np.empty(num_points-n_delete, dtype=object)
    newX = np.empty(n_delete, dtype=float).reshape(-1,1)
    newY = np.empty(n_delete, dtype=float)
    yReg = np.empty(num_points-n_delete, dtype=float)
    minY = np.empty(num_points-n_delete, dtype=float)
    maxY = np.empty(num_points-n_delete, dtype=float)


    setX = np.random.uniform(low=0.0, high=1.0, size = num_points)
    setY = np.random.uniform(low=0.0, high=1.0, size = num_points)
    # P= latin_hypercube_2d_uniform(num_points)

    setXYP[:,0] = setX
    setXYP[:,1] = setY
    
    for k in range (1,n_generations+1):
        #print("Generation: ",k)
  

        results = Parallel(n_jobs=num_processes)(delayed(pen)(setXYP[i,0], setXYP[i,1]) for i in range(num_points))
        setXYP[:,2] = np.fromiter(results,dtype=float)
        setXYP = setXYP[setXYP[:,2].argsort()] 

        newX = setXYP[:n_delete-1,0]
        newY = setXYP[:n_delete-1,1]


        newX = newX.reshape(-1,1)

        lowXpoint = np.min(newX)
        highXpoint = np.max(newX)

        transforming.fit(newX)

        newX_x = transforming.transform(newX)

        polyReg = LinearRegression().fit(newX_x, newY) #this finds all of the values
        
        
        coef1 = polyReg.coef_[1]
        coef2 = polyReg.coef_[0]
        yInt = polyReg.intercept_
        
        def poly(x):
            return coef1*math.pow(x,2) + coef2*x + yInt
        polyLine = list(map(poly, newX))


        minXBound = minSearch(lowXpoint)
        maxXBound = maxSearch(highXpoint)


        setXYP[n_delete:num_points,0] = np.random.uniform(low = minXBound, high = maxXBound, size = num_points-n_delete)


        yreg = map(poly, setXYP[n_delete:num_points,0])
        yReg = np.fromiter(yreg,dtype=float)

        min_y = map(minSearch, yReg)
        minY = np.fromiter(min_y,dtype=float)

        max_y = map(maxYSearch, yReg)
        maxY = np.fromiter(max_y,dtype=float)

        newY = map(randomY, minY, maxY)
        setXYP[n_delete:num_points,1] = np.fromiter(newY,dtype=float)

        num += 1
        exitLoop = exitCon(setXYP[:,2])
        if exitLoop == True:
            break
    return setXYP


nubmer_points_list= [100, 200,300,400,500,600,700,800,900,1000,1100,1200, 1300,1400,1500]
# n_runs= [5, 10, 15, 20, 25, 30, 35, 40, 50, 55, 60]
n_runs= [10]

def run_with_diff_n_runs(perc, num_points):
    final_res= []
    for n_r in n_runs:
        temp_res= []
        
        for i in range(n_r):
            res= run_example(perc, num_points)
            res= res[:,:-1]
            temp_res.extend(res)
            res= np.array(res)
            np.savetxt(
                f'./percent/{perc}/Ex3/N_{num_points}/'
                f'{i+1}_solns_run_{n_r}_{num_points}pts.txt',
                res,
                delimiter=','
            )
            ## Get distinct points
            #num, distinct_points = count_repeated_points(res)
            #distinct_points= np.array(distinct_points)
            # np.savetxt('./percent/{}N_'+str(num_points).format(perc)+"/Ex1"+"/"+str(i+1)+"_"+"solns"+'_'+'run_'+str(n_r)+'_'+str(num_points)+'pts'+'.txt', res, delimiter=',')

            
        final_res.append(temp_res)
            
    return final_res

# results = Parallel(n_jobs=num_processes)(delayed(run_with_diff_n_runs)(num_points) for num_points in nubmer_points_list)

import os
from joblib import Parallel, delayed

# Percentages to test
percent_list = list(range(10, 101, 10))

# Number of points
nubmer_points_list= [
    100,200,300,400,500,600,700,800,
    900,1000,1100,1200,1300,1400,1500
]

# Number of parallel runs
num_processes = -1  # same as in your code
for perc in percent_list:
    print(f"Running experiments for {perc}% Selection")
    Parallel(n_jobs=num_processes)(delayed(run_with_diff_n_runs)(perc, num_points) for num_points in nubmer_points_list)