from scipy.stats import skewnorm
import pandas as pd
import numpy as np
import warnings
import random
warnings.filterwarnings('ignore')
from geopy.distance import geodesic

def calculate_distances_in_meters(coords):
    num_points = len(coords)
    distance_matrix = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(num_points):
            if i != j:
                distance_matrix[i][j] = geodesic(coords[i], coords[j]).meters/1600
    return distance_matrix
