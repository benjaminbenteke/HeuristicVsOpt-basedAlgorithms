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
    path = f"./Ex1No/N_{num_points}"
    os.makedirs(path, exist_ok=True)

def regressLine(xVal, m, yInt): #returns the y value of any x coordinate on the line
  return m * xVal + yInt

linReg = LinearRegression()

def randomY(maxY, minY):
  return random.uniform(minY, maxY)

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
def mult_reg(x,z):
  return m1*x + m2*z + yInt

linReg = linear_model.LinearRegression()

transforming = PolynomialFeatures(degree=2, include_bias = False)
def poly(x,y):
  return coef1*math.pow(x,2) + coef2*x +coef1*math.pow(y,2) + coef2*y+yInt

linReg = linear_model.LinearRegression()


def regressLine(xVal, yInt): 
  return m * xVal + yInt


def randomY(maxY, minY):
  return random.uniform(minY, maxY)


def minxSearch(a):
  return a - (a - 0.0)*(math.pow(0.99, num))

def minySearch(b):
  return b - (b - 0.0)*(math.pow(0.99,num))

def minzSearch(c):
  return c - (c - 0.0)*(math.pow(0.99,num))


def maxxSearch(a):
  return a + (100/3.25 - a)*(math.pow(0.99,num))

def maxySearch(b):
  return b + (100/1.5625 - b)*(math.pow(0.99,num))

def maxzSearch(c):
  return c + (100/4.125 - c)*(math.pow(0.99,num))


def pen(a, b, c):

  def obj1(x):
    return (0.01*x + 0.01*(x + b + c) - 2.9)*x
  def obj2(x):
    return (0.05*x + 0.01*(a + x + c) - 2.88)*x
  def obj3(x):
    return (0.01*x + 0.01*(a + b + x) - 2.85)*x


  def con1_1(x):
    return 100-3.25*x-1.25*b-4.125*c
  def con1_2(x):
    return 100-2.29115*x-1.5625*b-2.8125*c

  def con2_1(x):
    return 100-3.25*a-1.25*x-4.125*c
  def con2_2(x):
    return 100-2.29115*a-1.5625*b-2.8125*c

  def con3_1(x):
    return 100-3.25*a-1.25*b-4.125*x
  def con3_2(x):
    return 100-2.29115*a-1.5625*b-2.8125*x

  b1 = [(0.0,100/3.25)]
  b2 = [(0.0, 100/1.5625)]
  b3 = [(0.0, 100/4.125)]

  constraint1 = {'type': 'ineq', 'fun': con1_1}
  constraint2 = {'type': 'ineq', 'fun': con1_2}
  constraint3 = {'type': 'ineq', 'fun': con2_1}
  constraint4 = {'type': 'ineq', 'fun': con2_2}
  constraint5 = {'type': 'ineq', 'fun': con3_1}
  constraint6 = {'type': 'ineq', 'fun': con3_2}


  cons1 = [constraint1, constraint2]
  cons2 = [constraint3, constraint4]
  cons3 = [constraint5, constraint6]

  solution1 = minimize(obj1,0.0,method='SLSQP', bounds= b1, constraints= cons1)
  solution2 = minimize(obj2,0.0,method='SLSQP', bounds= b2, constraints= cons2)
  solution3 = minimize(obj3,0.0,method='SLSQP', bounds= b3, constraints= cons3)


  shadowX = solution1.x[0] 
  shadowY = solution2.x[0] 
  shadowZ = solution3.x[0] 

  penalty = math.sqrt((math.pow(a-shadowX,2))+(math.pow(b-shadowY,2))+(math.pow(c-shadowZ,2)))

  return penalty


def generate_random_value(yMinSearch,yMaxSearch,i):
    return np.random.uniform(yMinSearch[i], yMaxSearch[i])



num_points= 200 #len(grid_points)
n_generations= 1000
def run_example(num_points):
    
    def mult_reg(x,z):
        return m1*x + m2*z + yInt

    linReg = linear_model.LinearRegression()

    transforming = PolynomialFeatures(degree=2, include_bias = False)
    def poly(x,y):
      return coef1*math.pow(x,2) + coef2*x +coef1*math.pow(y,2) + coef2*y+yInt

    linReg = linear_model.LinearRegression()


    def regressLine(xVal, yInt): 
      return m * xVal + yInt


    def randomY(maxY, minY):
      return random.uniform(minY, maxY)


    def minxSearch(a):
      return a - (a - 0.0)*(math.pow(0.99, num))

    def minySearch(b):
      return b - (b - 0.0)*(math.pow(0.99,num))

    def minzSearch(c):
      return c - (c - 0.0)*(math.pow(0.99,num))


    def maxxSearch(a):
      return a + (100/3.25 - a)*(math.pow(0.99,num))

    def maxySearch(b):
      return b + (100/1.5625 - b)*(math.pow(0.99,num))

    def maxzSearch(c):
      return c + (100/4.125 - c)*(math.pow(0.99,num))

    # initializing all of the points
    
    m= int((25*num_points)/100) #30 # number of points to select
    n_points= num_points

    setXYZP = np.zeros((n_points,4))

    setX = np.empty(n_points, dtype=object)
    setY = np.empty(n_points, dtype=object)
    setZ = np.empty(n_points, dtype=object)

    newX = np.empty(m, dtype=float).reshape(-1,1)
    newY = np.empty(m, dtype=float)
    newZ = np.empty(m, dtype=float).reshape(-1,1)

    setX = np.random.uniform(low=0.0, high= 100/3.25, size = n_points) #14, 15, 30
    setY = np.random.uniform(low=0.0, high= 100/1.5625, size = n_points)
    setZ = np.random.uniform(low=0.0, high= 100/4.125, size = n_points)

    setXYZP[:,0] = setX
    setXYZP[:,1] = setY
    setXYZP[:,2] = setZ

    num= 1

    for k in range (1,n_generations+1):
#       print("Generation ", k)

      results = Parallel(n_jobs=-1,prefer= 'threads')(delayed(pen)(setXYZP[i,0], setXYZP[i,1], setXYZP[i,2]) for i in range(n_points))


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
      currentY= list(map(randomY, minY, maxY))
      setXYZP[m:n_points,1] =  currentY# newY

      # Regression for z
      linReg= linear_model.LinearRegression()
      linReg.fit(newX.reshape(-1, 1), newZ)

      l= linReg.coef_[0]
      yInt = linReg.intercept_

    #   minY = list(map(minySearch, newY))
    #   maxY = list(map(maxySearch, newY))

      zreg = list(map(regLine, setXYZP[m:n_points,0]))

      minZ = list(map(minzSearch, zreg))
      maxZ = list(map(maxzSearch, zreg))

      newZ = list(map(randomY, minZ, maxZ))

      setXYZP[m:n_points,2] = newZ

#       if num % 10 == 0:
#         newGraph = plt.axes(projection ="3d")
#         newGraph.scatter3D(setXYZP[m:,0],setXYZP[m:,1],setXYZP[m:,2])
#         newGraph.scatter3D(setXYZP[:m-1,0],setXYZP[:m-1,1],setXYZP[:m-1,2], color = 'r')
#         newGraph.set_xlabel('X-axis', fontweight ='bold')
#         newGraph.set_ylabel('Y-axis', fontweight ='bold')
#         newGraph.set_zlabel('Z-axis', fontweight ='bold')
#         plt.show()
      num += 1

#       exitLoop = exitCon(setXYZP[:,3])

#       if exitLoop == True:
#         break
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
    print(num_points)
    results = Parallel(n_jobs=-1,prefer= 'threads')(delayed(run_each)(num_points) for _ in range(n_r))
    for i in range(n_r):
        np.savetxt('./Ex1No/N_'+str(num_points)+"/"+str(i+1)+""+"solns"+''+'run_'+str(n_r)+'_'+str(num_points)+'pts'+'.txt', results[i], delimiter=',')
    #return results


def run_with_diff_n_runs(num_points):
    final_res= []
    
    ##results = 
    Parallel(n_jobs=-1,prefer= 'threads')(delayed(run_parallel)(num_points,n_r) for n_r in n_runs)
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
# Parallel(n_jobs=-1,prefer= 'threads')(delayed(run_with_diff_n_runs)(num_points) for num_points in nubmer_points_list)
# Parallel(n_jobs=-1,prefer= 'threads')(delayed(run_parallel)(num_points,10) for num_points in nubmer_points_list)
run_parallel(num_points= 500, n_r= 10)