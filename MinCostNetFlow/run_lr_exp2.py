import pandas as pd
import numpy as np
from gurobipy import Model, GRB
from gurobipy import quicksum
import gurobipy as gp
import sys
import os
import pickle

parent_dir = os.path.abspath(os.path.join(os.getcwd(), 'UAM_system_optimization/'))
os.chdir('UAM_system_optimization/MinCostNetFlow/')
sys.path.append(parent_dir)
from model.schedule import ScheduleGenerator
from model.topology import *
from model.NetworkUtils import * 
from model.LR import LagrangianRelaxedFleetSize as LRFS
os.environ['GRB_LICENSE_FILE'] = '/home/albert/gurobipy/gurobi.lic'


np.random.seed(9)
gen = ScheduleGenerator('../data/LAX_ind.csv', '../data/T_F41SCHEDULE_B43.csv')
schedule, pax_arrival_times, num_pax_per_flight = gen.get_one_day(8, 1, directional_demand=5000)
vertiports = ['LAX', 'DTLA', 'LGB', 'WDHL', 'UVS', 'ANH', 'HWD', 'PSD', 'BVH']
vertiport_dict = {idx: i for idx, i in enumerate(vertiports)}
schedule['schedule'] = np.ceil(schedule['schedule']/5)
schedule = schedule.groupby(['schedule', 'od']).size().reset_index(name='count')
apt_dt = schedule[schedule['od'] == 'LAX_DTLA'].reset_index(drop=True)
dt_apt = schedule[schedule['od'] == 'DTLA_LAX'].reset_index(drop=True)

# Uniform distribution
p = np.repeat(1,len(vertiports)-1) / (len(vertiports)-1)
np.random.seed(9)
demand_dict, flight_count = get_demand_dict(apt_dt, dt_apt, p.cumsum(), vertiports)

# Distribution from https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9441659
p_2 = np.array([6.1, 3.3, 1.5, 1.5, 5.4, 4.3, 2.5, 2.2])
p_2 = p_2 / p_2.sum()
np.random.seed(9)
demand_dict_2, flight_count_2 = get_demand_dict(apt_dt, dt_apt, p_2.cumsum(), vertiports)

# Distribution with epislon
p_3 = modify_demand_dist(p_2, epsilon = 0.5)
np.random.seed(9)
demand_dict_3, flight_count_3 = get_demand_dict(apt_dt, dt_apt, p_3.cumsum(), vertiports)

coordinates = np.array([(33.94417072663332, -118.40248449470265),
                        (34.043119432990345, -118.26720629762752),
                        (33.81606030845284, -118.15122129260827),
                        (34.17160155814583, -118.60525790222589),
                        (34.13843197335217, -118.35511311047983),
                        (33.812102161009804, -117.91897753216998),
                        (34.09174681098575, -118.32725521957076),
                        (34.08807589928915, -118.55063573741867),
                        (34.06666969690935, -118.4114352069469)])
                        
soc_transition_time = np.array([0.0129,0.0133,0.0137,0.0142,0.0147,
                        0.0153,0.0158,0.0166,0.0172,0.018,
                        0.0188,0.0197,0.0207,0.0219,0.0231,
                        0.0245,0.026,0.0278,0.03,0.0323,
                        0.0351,0.0384,0.0423,0.0472,0.0536,
                        0.0617,0.0726,0.0887,0.1136,0.1582,
                        0.2622,0.9278,])*60

distance_matrix = calculate_distances_in_meters(coordinates)
flight_time = np.zeros(shape=distance_matrix.shape)
for i in range(distance_matrix.shape[0]):
    for j in range(distance_matrix.shape[1]):
        distance = distance_matrix[i][j]
        cruise_speed = 180/12 # miles per 5 minutes
        if i != j:
            flight_time[i][j] = 2 + np.ceil(distance / cruise_speed)

energy_consumption = np.zeros(shape=distance_matrix.shape)
for i in range(distance_matrix.shape[0]):
    for j in range(distance_matrix.shape[1]):
        distance = distance_matrix[i][j]
        if i!= j:
            if distance <= 20:
                energy_consumption[i][j] = distance * 1 + 5
            else:
                energy_consumption[i][j] = 20 + (distance - 20) * 0.65 + 5
energy_consumption = energy_consumption / 160 * 100

charging_time = np.zeros(shape=distance_matrix.shape)
for i in range(distance_matrix.shape[0]):
    for j in range(distance_matrix.shape[1]):
        distance = distance_matrix[i][j]
        if i!= j:
            charging_time[i][j] = soc_transition_time[:int(np.ceil(energy_consumption[i][j]/2.5))].sum()
charging_time = charging_time.astype(int)

od_matrix = np.array([[0,1,1,1,1,1,1,1,1],
                      [1,0,1,1,1,1,1,1,1],
                      [1,1,0,1,1,1,1,1,1],
                      [1,1,1,0,1,1,1,1,1],
                      [1,1,1,1,0,1,1,1,1],
                      [1,1,1,1,1,0,1,1,1],
                      [1,1,1,1,1,1,0,1,1],
                      [1,1,1,1,1,1,1,0,1],
                      [1,1,1,1,1,1,1,1,0]])
flight_time = flight_time.astype(int)
energy_consumption = np.ceil(energy_consumption / 2.5).astype(int)

network2 = ChargingNetwork(vertiports, flight_time, energy_consumption, od_matrix)
nodes, supply, edges, cost, c = network2.populate_network()

model_name = 'STAR_8_0721'

def get_flight_edges(i, model_param, flight_time=flight_time, energy_consumption=energy_consumption):
    K = 32
    v1, v2, t = i
    flight_time_ij = flight_time[v1, v2]
    soc_level_drop = int(energy_consumption[v1, v2])

    volume = quicksum(model_param[(v1, t, k), (v2, int(t+flight_time_ij), k-soc_level_drop)] for k in range(soc_level_drop, K+1))
    
    return volume

def main():
    optimizer = LRFS(model_path = f'../output/star_network/LR/{model_name}.mps',
                    edges = edges,
                    cost = c, 
                    flight_time = flight_time,
                    energy_consumption = energy_consumption,
                    flight_demand=demand_dict_2)


    optimizer.solve(path='../output/star_network/network_flow_0721/exp2_lr.pkl',
                    max_iter=1000)

if __name__ == '__main__':
    main()


