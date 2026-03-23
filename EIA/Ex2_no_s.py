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

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import StrMethodFormatter
import joblib
from joblib import Parallel, delayed
joblib.cpu_count()
num_processes = -1
# np.set_printoptions(precision=1064)
np.set_printoptions(suppress=True)

import os

for num_points in range(100, 1501, 100):
    path = f"./Ex2No/N_{num_points}"
    os.makedirs(path, exist_ok=True)





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

import joblib
from joblib import Parallel, delayed
joblib.cpu_count()
num_processes = -1

# Ex1
# global m1,m2,m3
# m1,m2,m3= None, None, None
## Ex2 (without True solution set)

def mult_reg(x,z):
  return m1*x + m2*z + m3*(x**2) + m4*x*z + m5*(z**2) + yInt

transforming = PolynomialFeatures(degree=2, include_bias = False)
linReg = LinearRegression()

def randomY(maxY, minY):
  return random.uniform(minY, maxY)


def minxSearch(a1):
  return a1 - (a1 - 0)*(math.pow(0.99,num))

def minySearch(b1):
  return b1 - (b1 - 0)*(math.pow(0.99,num))

def minzSearch(c1):
  return c1 - (c1 - 0)*(math.pow(0.99,num))

def maxxSearch(a1):
  return a1 + (1 - a1)*(math.pow(0.99,num))

def maxySearch(b1):
  return b1 + (1 - b1)*(math.pow(0.99,num))

def maxzSearch(c1):
  return c1 + (1 - c1)*(math.pow(0.99,num))


def pen(a, b, c):

  def func1(x):
    return (x)**2 - x*b - x

  def func2(x):
    return (x)**2 - 0.5*a*x - 2*x

  def func3(x):
    return (x - 0.5)**2

  b2 = np.min([math.sqrt(abs(1 - (a)**2)), 1 - c])
  bound1 = [(0.0,math.sqrt(abs(1.0-(b)**2)))]
  bound2 = [(0.0,1.0)]
  bound3 = [(0.0, abs(1.0 - b))]

  def con1(x):
    return 1 - a**2 - x**2

  def con2(x):
    return 1 - c - x

  constraint1 = {'type': 'ineq', 'fun': con1}
  constraint2 = {'type': 'ineq', 'fun': con2}

  cons = [constraint1, constraint2]

  result1 = minimize(func1, x0=0.0, method='SLSQP', bounds = bound1)#, options={'maxiter':5})
  result2 = minimize(func2, x0=0.0, method='SLSQP', bounds = bound2, constraints = cons)#, options={'maxiter':5})
  result3 = minimize(func3, x0=0.0, method='SLSQP', bounds = bound3)# options={'maxiter':5})

  shadowX = result1.x
  shadowY = result2.x
  shadowZ = result3.x

  penalty = math.sqrt((math.pow(a-shadowX,2))+(math.pow(b-shadowY,2))+(math.pow(c-shadowZ,2)))

  return penalty



def generate_random_value(yMinSearch,yMaxSearch,i):
    return np.random.uniform(yMinSearch[i], yMaxSearch[i])



num_points= 200 #len(grid_points)
n_generations= 1000
def run_example(num_points):
    
    def mult_reg(x,z):
        return m1*x + m2*z + m3*(x**2) + m4*x*z + m5*(z**2) + yInt

    transforming = PolynomialFeatures(degree=2, include_bias = False)
    linReg = LinearRegression()

    def randomY(maxY, minY):
        return random.uniform(minY, maxY)


    def minxSearch(a1):
        return a1 - (a1 - 0)*(math.pow(0.99,num))

    def minySearch(b1):
        return b1 - (b1 - 0)*(math.pow(0.99,num))
 
    def minzSearch(c1):
        return c1 - (c1 - 0)*(math.pow(0.99,num))

    def maxxSearch(a1):
        return a1 + (1 - a1)*(math.pow(0.99,num))

    def maxySearch(b1):
        return b1 + (1 - b1)*(math.pow(0.99,num))

    def maxzSearch(c1):
        return c1 + (1 - c1)*(math.pow(0.99,num))

    # initializing all of the points
    
    m= int((25*num_points)/100) #30 # number of points to select
    n_points= num_points

    n_points= num_points
    setXYZP = np.zeros(n_points*4)
    setXYZP = setXYZP.reshape(n_points,4)

    setX = np.empty(n_points, dtype=object)
    setY = np.empty(n_points, dtype=object)
    setZ = np.empty(n_points, dtype=object)

    newX = np.empty(n_points-m, dtype=float).reshape(-1,1)
    newY = np.empty(n_points-m, dtype=float)
    newZ = np.empty(n_points-m, dtype=float).reshape(-1,1)
    
    setX = np.random.uniform(low=0.0, high=1.0, size = n_points) #14, 15, 30
    setY = np.random.uniform(low=0.0, high=1.0, size = n_points)
    setZ = np.random.uniform(low=0.0, high=1.0, size = n_points)

    setXYZP[:,0] = setX
    setXYZP[:,1] = setY
    setXYZP[:,2] = setZ

    num= 1

    for k in range (1,n_generations+1):
        
        results = Parallel(n_jobs=num_processes)(delayed(pen)(setXYZP[i,0], setXYZP[i,1], setXYZP[i,2]) for i in range(n_points))


        setXYZP[:,3]= np.fromiter(results,dtype=float)
        setXYZP= setXYZP[setXYZP[:,3].argsort()]


        newX = setXYZP[:m-1,0]
        newY = setXYZP[:m-1,1]
        newZ = setXYZP[:m-1,2]

        lowXpoint = np.min(newX)
        highXpoint = np.max(newX)

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
        currentY= list(map(randomY, minY, maxY))
        setXYZP[m:n_points,1] =  currentY# newY

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

    return setXYZP


nubmer_points_list= [100, 200,300,400,500,600,700,800,900,1000,1100,1200, 1300,1400,1500]
# n_runs= [5, 10, 15, 20, 25, 30, 35, 40, 50, 55, 60]
n_runs= [10] #[5, 10, 15, 20]

def run_each(num_points):
    res= run_example(num_points)
    res= res[:,:-1]
    #temp_res.extend(res)
    res= np.array(res)#
    ## Get distinct points
    #num, distinct_points = count_repeated_points(res)
    #distinct_points= np.array(distinct_points)
    
    return res

def run_parallel(num_points,n_r):
    results = Parallel(n_jobs=num_processes)(delayed(run_each)(num_points) for _ in range(n_r))
    for i in range(n_r):
        np.savetxt('./Ex2No/N_'+str(num_points)+"/"+str(i+1)+""+"solns"+''+'run_'+str(n_r)+'_'+str(num_points)+'pts'+'.txt', results[i], delimiter=',')
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
            #np.savetxt('./EIA/solns_runs/Ex5/N_'+str(num_points)+'/'+"n_runs"+""+str(n_r)+"/"+str(i+1)+""+"solns"+''+'run'+str(n_r)+'_'+str(num_points)+'pts'+'.txt', distinct_points, delimiter=',')

            
        #final_res.append(temp_res)
            
    #return results

#results = Parallel(n_jobs=num_processes)(delayed(run_parallel)(num_points,10) for num_points in nubmer_points_list)



run_parallel(num_points= 1500,n_r= 10)