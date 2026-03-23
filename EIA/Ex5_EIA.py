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

import os

for num_points in range(100, 1501, 100):
    path = f"./Ex5/N_{num_points}"
    os.makedirs(path, exist_ok=True)
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


## Ex5
def pen(a, b, c):

  def obj1(x):
    return x[0]**2 + x[0]*x[1] + x[1]**2 + (x[0] + x[1])*c - 25*x[0] - 38*x[1]
  def obj2(x):
    return x**2 + (a + b)*x - 25*x


  def con1_1(x):
    return 14-x[0] - 2*x[1] + c

  def con2_1(x):
    return 30-3*x[0]  - 2*x[1] - c


  def con1_2(x):
    return 14-a  - 2*b + x

  def con2_2(x):
    return 30-3*a  - 2*b - x

  x0 = (0.0,0.0)

  b1 = (0.0,8.0)
  b2 = (3.0,11.0)
  b3 = [(0.0,8.0)]


  bnds1 = (b1,b2)

  constraint1 = {'type': 'ineq', 'fun': con1_1}
  constraint2 = {'type': 'ineq', 'fun': con2_1}

  cons1 = [constraint1, constraint2]

  solution = minimize(obj1,x0,method='SLSQP', bounds = bnds1, constraints = cons1)#, options={'maxiter':5})

  shadowX = solution.x[0]
  shadowY = solution.x[1]

  constraint12 = {'type': 'ineq', 'fun': con1_2}
  constraint22 = {'type': 'ineq', 'fun': con2_2}

  cons2 = [constraint12, constraint22]

  solution2 = minimize(obj2,0.0,method='SLSQP', bounds = b3, constraints = cons2)#, options={'maxiter':5})

  shadowZ = solution2.x

  penalty = math.sqrt((math.pow(a-shadowX,2))+(math.pow(b-shadowY,2))+(math.pow(c-shadowZ,2)))

  return penalty

n_generations= 1000
def run_example(num_points):
    num= 1
    
    def mult_reg(x,z):
        return m1*x + m2*z + yInt
    def randomY(maxY, minY):
      return random.uniform(minY, maxY)

    def minxSearch(a):
      return a - (a - 0.0)*(math.pow(0.99, num)) #//5 + 1

    def minySearch(b):
      return b - (b - 3.0)*(math.pow(0.99,num))

    def minzSearch(c):
      return c - (c - 0.0)*(math.pow(0.99,num))


    def maxxSearch(a):
      return a + (8.0 - a)*(math.pow(0.99,num))

    def maxySearch(b):
      return b + (11.0 - b)*(math.pow(0.99,num))

    def maxzSearch(c):
      return c + (8.0 - c)*(math.pow(0.99,num))
    n_points= num_points
    m= int((25*n_points)/100)
    setXYZP = np.zeros((n_points,4))

    setX = np.empty(n_points, dtype=object)
    setY = np.empty(n_points, dtype=object)
    setZ = np.empty(n_points, dtype=object)

    newX = np.empty(m, dtype=float).reshape(-1,1)
    newY = np.empty(m, dtype=float)
    newZ = np.empty(m, dtype=float).reshape(-1,1)
    setXYZP.shape
    noise = np.random.uniform(low=0.0, high=0.001, size = n_points-m)
    
    ## Regression step: x->y and x->z
    setX = np.random.uniform(low=0.0, high= 8.0, size = n_points) #14, 15, 30
    setY = np.random.uniform(low=3.0, high= 11.0, size = n_points)
    setZ = np.random.uniform(low=0.0, high= 8.0, size = n_points)

    setXYZP[:,0] = setX
    setXYZP[:,1] = setY
    setXYZP[:,2] = setZ

    
    
    for k in range (1,n_generations+1):
        results = Parallel(n_jobs=num_processes)(delayed(pen)(setXYZP[i,0], setXYZP[i,1], setXYZP[i,2]) for i in range(n_points))


        setXYZP[:,3]= np.fromiter(results,dtype=float)
        setXYZP= setXYZP[setXYZP[:,3].argsort()]


        newX = setXYZP[:m-1,0]
        newY = setXYZP[:m-1,1]
        newZ = setXYZP[:m-1,2]

        lowXpoint = np.min(newX)
        highXpoint = np.max(newX)

        # Regression for y
        linReg = LinearRegression()

        linReg.fit(newX.reshape(-1, 1), newY)

        def regLine(xVal):
            return l * xVal + yInt

        l= linReg.coef_
        yInt = linReg.intercept_

        minXBound = minxSearch(lowXpoint)
        maxXBound = maxxSearch(highXpoint)

        setXYZP[m:n_points,0] = np.random.uniform(low = minXBound, high = maxXBound, size = n_points-m)

        yreg = list(map(regLine, setXYZP[m:n_points,0]))
        yReg = np.fromiter(yreg,dtype=float)

        minY = list(map(minySearch, yReg))
        maxY = list(map(maxySearch, yReg))

        setXYZP[m:n_points,1] = list(map(randomY, minY, maxY)) # newY

        # Regression for z
        linReg= linear_model.LinearRegression()
        linReg.fit(newX.reshape(-1, 1), newZ)

        l= linReg.coef_[0]
        yInt = linReg.intercept_


        zreg = list(map(regLine, setXYZP[m:n_points,0]))

        minZ = list(map(minzSearch, zreg))
        maxZ = list(map(maxzSearch, zreg))

        newZ = list(map(randomY, minZ, maxZ))

        setXYZP[m:n_points,2] = newZ

        
        num += 1

        exitLoop = exitCon(setXYZP[:,3])

        if exitLoop == True:
            break
            
    return setXYZP


nubmer_points_list= [100, 200, 300, 400,500,600,700,800,900,1000,1100,1200, 1300,1400,1500]
n_runs= [10]

def run_each(num_points):
    res= run_example(num_points)
    res= res[:,:-1]
    res= np.array(res)#
 
    return res

def run_parallel(num_points,n_r):
    results = Parallel(n_jobs=num_processes)(delayed(run_each)(num_points) for _ in range(n_r))
    for i in range(n_r):
        np.savetxt('./Ex5/N_'+str(num_points)+"/"+str(i+1)+"_"+"solns"+'_'+'run_'+str(n_r)+'_'+str(num_points)+'pts'+'.txt', results[i], delimiter=',')
    #return results


def run_with_diff_n_runs(num_points):
    final_res= []
    
    ##results = 
    Parallel(n_jobs=num_processes)(delayed(run_parallel)(num_points,n_r) for n_r in n_runs)
    #for n_r in n_runs:
        #temp_res= []
        
        #for i in range(n_r):
            #res= run_example(num_points)#
            #res= res[:,:-1]
            #temp_res.extend(res)
            #res= np.array(res)
            ## Get distinct points
            #num, distinct_points = count_repeated_points(res)
            #distinct_points= np.array(distinct_points)
            #np.savetxt('./EIA/solns_runs/Ex5/N_'+str(num_points)+'/'+"n_runs"+"_"+str(n_r)+"/"+str(i+1)+"_"+"solns"+'_'+'run_'+str(n_r)+'_'+str(num_points)+'pts'+'.txt', distinct_points, delimiter=',')

            
        #final_res.append(temp_res)
            
    #return results

results = Parallel(n_jobs=num_processes)(delayed(run_with_diff_n_runs)(num_points) for num_points in nubmer_points_list)