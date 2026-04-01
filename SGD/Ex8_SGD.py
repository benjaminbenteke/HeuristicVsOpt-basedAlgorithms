import numpy as np
from scipy.optimize import minimize
import pandas as pd
import itertools

import numpy as np
from numpy import random

import random
import math
import time
import sys
 
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint
from scipy import stats
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures
from scipy.spatial.distance import directed_hausdorff

import joblib
from joblib import Parallel, delayed
joblib.cpu_count()
num_processes= -1
np.set_printoptions(suppress=True)

import os

for num_points in range(100, 1501, 100):
    path = f"./Ex8/N_{num_points}"
    os.makedirs(path, exist_ok=True)

def randomY(maxY, minY):
  return random.uniform(minY, maxY)

def newLine(xVal): #returns the y value of any x coordinate on the line
  return (-1 * xVal) + 1

eps_t= 1e-5
delta= 1e-2 

def listComplementElements(list1, list2):
    storeResults = []

    for num in list1:
        if num not in list2: # this will essentially iterate your list behind the scenes
            storeResults.append(num)

    return storeResults

def exitCon(pen):
    return np.all(pen<eps_t)

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

#https://www.youtube.com/watch?v=VVEbwuyoWkU
def gold(func, xl, xu): # OK
    
  ea = 100
  i = 1

  ratio = (5 ** 0.5 - 1) / 2
  d = ratio * (xu - xl)
  x1 = xl + d
  x2 = xu - d

  f1 = func(x1)
  f2 = func(x2)

  while ea > 0.1:
    if f1 < f2:
      xl = x2
      x2 = x1
      f2 = f1
      x1 = xl + ratio*(xu-xl)
      f1 = func(x1)
    else:
      xu = x1
      x1 = x2
      f1 = f2
      x2 = xu - ratio*(xu-xl)
      f2 = func(x2)

    if f1 < f2:
      xopt= x1
    else:
      xopt= x2
    
    ea = (1 - ratio) * abs((xu - xl) / xopt) * 100
    i += 1

  return xopt


def pen(a,b):

  def func1(x):
    return -x

  def func2(x):
    return -x

  def g11(x):
    return 1-x-b
  
  def g12(x):
    return 0-(-2*x+b)

  def g21(x):
    return 1-x-a

  def g22(x):
    return 0-(x-2*a)

    
  constraints1 = [{'type': 'ineq', 'fun': g11},{'type': 'ineq', 'fun': g12}]

  # bound1 = [(0.0,1.0-b)]
  # bound2 = [(0.0,1.0-a)]

#   constraint1 = {'type': 'ineq', 'fun': g1}
  constraint2 = [{'type': 'ineq', 'fun': g21},{'type': 'ineq', 'fun': g22}]
  
#   bounds = bound1
  result1 = minimize(func1, x0=0.0, method='SLSQP', constraints= constraints1) #, options={'maxiter':5}
  result2 = minimize(func2, x0=0.0, method='SLSQP',constraints= constraint2)

  shadowX = result1.x
  shadowY = result2.x

  penalty = math.sqrt((math.pow(a-shadowX,2))+(math.pow(b-shadowY,2)))
  return penalty

def pen_children(p,sigma,P):
    children= sigma[:,p].reshape(-1,1)+P[p,:-1]
    penK= Parallel(n_jobs=-1,prefer= 'threads')(delayed(pen)(children[i][0], children[i][1]) for i in range(n_children))
    return penK

def transformation(p_pen,sigma, index_P,pen_ch):
    return sigma[:,index_P]@(p_pen-pen_ch[:,index_P])/n_children

def get_sig(num):
    return np.random.normal(loc = 0.0, scale = (0.99 ** num)/2, size= n_children)


def get_beta(P, sigma, penchildren):
    beta= []
    for p in range(n_parents):
        beta.append(transformation(P[p,-1], sigma, p,penchildren))
    return np.array(beta) 

def run_example(n_parents):
    
    n_children= 25 # per parent
    n_generations= 1000
    n= 2
    xmin, xmax, ymin, ymax= 0.0, 1.0, 0.0, 1.0
    P = np.empty((n_parents, n+1))
    P[:, 0:2] = np.random.uniform(low= xmin, high= xmax, size = (n_parents,n))
    
    def pen_children(p,sigma,P):
        children= sigma[:,p].reshape(-1,1)+P[p,:-1]
        penK= Parallel(n_jobs=-1,prefer= 'threads')(delayed(pen)(children[i][0], children[i][1]) for i in range(n_children))
        return penK

    def transformation(p_pen,sigma, index_P,pen_ch):
        return sigma[:,index_P]@(p_pen-pen_ch[:,index_P])/n_children

    def get_sig(num):
        return np.random.normal(loc = 0.0, scale = (0.99 ** num)/2, size= n_children)
    
    def get_beta(P, sigma, penchildren):
        beta= []
        for p in range(n_parents):
            beta.append(transformation(P[p,-1], sigma, p,penchildren))
        return np.array(beta) 

    sigma= None
    num= 1
    for k in range(1,n_generations+1):
        sigma= [get_sig(num) for _ in range(n_parents)]# map(get_sig, np.arange(n_parents))
        sigma= np.array(sigma).T


        penchildren = Parallel(n_jobs=-1,prefer= 'threads')(delayed(pen_children)(p,sigma,P) for p in range(n_parents))
        penchildren= np.array(penchildren).T
        P[:,-1] = Parallel(n_jobs=-1,prefer= 'threads')(delayed(pen)(P[p,:-1][0], P[p,:-1][1]) for p in range(n_parents))
        beta= get_beta(P, sigma, penchildren)
        P[:,:-1]+= np.array(beta).reshape(-1,1)

    
        num+= 1  
        
    sol1= np.vstack((P[:,0], P[:,1],P[:,2])).T
#     fit_= P[:,-1]<=0.1
#     solns= sol1[fit_]
        
    return sol1


def run_each(num_points):
    res= run_example(num_points)
    # res= res[:,:-1]
    #temp_res.extend(res)
    res= np.array(res)#
    ## Get distinct points
    #num, distinct_points = count_repeated_points(res)
    #distinct_points= np.array(distinct_points)

    return res

## Task 1
nubmer_points_list= [100, 200,300,400,500,600,700,800,900,1000,1100,1200, 1300,1400,1500]
# n_runs= [5, 10, 15, 20, 25, 30, 35, 40, 50, 55, 60]
n_runs= 10#[10] #[5, 10, 15, 20]
n_r= 10
def run_with_diff_n_runs(num_points):
    final_res= []
    print("************** ",num_points)
    for i in range(n_runs):
        res= run_example(num_points)
        # temp_res.extend(res)
        res= np.array(res)

        np.savetxt('./Ex8/N_'+str(num_points)+"/"+str(i+1)+"_"+"solns"+'_'+'run_'+str(n_r)+'_'+str(num_points)+'pts'+'.txt', res, delimiter=',')

            
        #final_res.append(temp_res)
            
    return final_res

results = Parallel(n_jobs=num_processes)(delayed(run_with_diff_n_runs)(num_points) for num_points in nubmer_points_list)