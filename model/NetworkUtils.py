from scipy.stats import skewnorm
import pandas as pd
import numpy as np
import warnings
import random
warnings.filterwarnings('ignore')
from geopy.distance import geodesic
from model.ScheduleUtils import inverse_cdf

def calculate_distances_in_meters(coords):
    num_points = len(coords)
    distance_matrix = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(num_points):
            if i != j:
                distance_matrix[i][j] = geodesic(coords[i], coords[j]).meters/1600
    return distance_matrix

def HHI(array):
    return np.sum(np.square(array))

def get_demand_dict(apt_dt, dt_apt, pcum, vertiports):
    demand_dict = {}
    flight_count = np.zeros(shape=(len(vertiports),))
    for idx, row in apt_dt.iterrows():
        time = row['schedule']
        demand = inverse_cdf(pcum, row['count'])
        od, count = np.unique(demand, return_counts=True)
        od += 1
        for destination, volume in zip(od, count):
            demand_dict[(0, destination, int(time))] = volume
            flight_count[destination] += volume
            
    for idx, row in dt_apt.iterrows():
        time = row['schedule']
        demand = inverse_cdf(pcum, row['count'])
        od, count = np.unique(demand, return_counts=True)
        od += 1
        for origin, volume in zip(od, count):
            demand_dict[(origin, 0, int(time))] = volume
            flight_count[origin] += volume

    return demand_dict, flight_count
    

def get_2_vertiport_demand(demand_dict, vertiport_index):
    APT_CBD = np.zeros(shape=(288,), dtype=int)
    CBD_APT = np.zeros(shape=(288,), dtype=int)
    for key, value in demand_dict.items():
        origin, destination, time = key
        if (origin == 0) & (destination == vertiport_index):
            APT_CBD[time-1] = value
        elif (destination == 0) & (origin == vertiport_index):
            CBD_APT[time-1] = value

    return (APT_CBD, CBD_APT)

def modify_demand_dist(p, epsilon):
    am = (p/np.median(p))**(1+epsilon)
    am = am/sum(am)
    return am

