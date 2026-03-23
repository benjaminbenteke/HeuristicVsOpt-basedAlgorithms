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

eps_t= 1e-5
delta= 1e-2

def listComplementElements(list1, list2):
    storeResults = []

    for num in list1:
        if num not in list2: # this will essentially iterate your list behind the scenes
            storeResults.append(num)

    return storeResults

def exitCon(pen):
    return np.all(pen<delta)

def euclidean_distance_matrix(array):
    # Calculate the Euclidean distance between each pair of elements in the array
    diff = array[:, np.newaxis] - array
    distance_matrix = np.sqrt(np.sum(diff**2, axis=-1))
    return distance_matrix
def row_indices_with_true(bool_matrix):
    # Find the row indices that contain at least one True value
    row_indices = np.where(np.any(bool_matrix, axis=1))[0]
    return row_indices

def count_repeated_points(array):
    distance_matrix = euclidean_distance_matrix(array)
    num_repeated_points = 0

    # Create a mask to identify repeated points based on distance < delta
    np.fill_diagonal(distance_matrix, np.inf)
    mask = distance_matrix < delta

    # Count the number of True values in the mask (excluding the diagonal elements)
#     num_repeated_points = np.sum(mask) - array.shape[0]
    mask= np.tril(mask)
#     mask_1= np.tril(mask_1)
    res= row_indices_with_true(mask)
    all_indices= np.arange(len(array))
    distinct_indices= listComplementElements(all_indices, res)
    return len(distinct_indices), array[distinct_indices]