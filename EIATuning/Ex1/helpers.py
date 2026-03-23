import os
import re
import numpy as np
import matplotlib. pyplot as plt

def extract_number(s):
    return int(re.search(r'\d+', s).group())

def list_subfolders(folder_path):
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    sorted_subfolders = sorted(subfolders, key=lambda x: int(x.split("_")[-1]))
    return sorted_subfolders


def list_txt_files(folder_path):
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
#     txt_files = sorted(txt_files, key=lambda x: int(x.split("_")[2]))
    return txt_files


eps_t= 1e-5
delta= 1e-2 #1e-2
delta

def calculate_distance(point1, point2):
    # Calculate the Euclidean distance between two points of any dimension
    return sum((x - y) ** 2 for x, y in zip(point1, point2)) ** 0.5
#     return sum(abs(x - y) for x, y in zip(point1, point2))

    
# def count_repeated_points(points,n_runs):
#     n= len(points)
#     ep= 1/(n_runs*(n-1))
#     # Initialize a list to store distinct points
#     distinct_points = []

#     # Iterate through each point in the set
#     for point in points:
#         # Check if the point is distinct from all previously considered distinct points
#         is_distinct = True
#         for distinct_point in distinct_points:
#             if calculate_distance(point, distinct_point) < ep:
#                 is_distinct = False
#                 break
#         # If the point is distinct, add it to the list of distinct points
#         if is_distinct:
#             distinct_points.append(point)
            
#     return len(distinct_points), distinct_points