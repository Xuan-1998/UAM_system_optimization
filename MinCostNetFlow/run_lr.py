import pandas as pd
import numpy as np
from gurobipy import Model, GRB
from gurobipy import quicksum
import gurobipy as gp
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
from IPython.display import clear_output
import pickle
from src.topo import AssignmentNetwork, FlightTask, ChargingNetwork
from tqdm import tqdm


DTLA = pd.read_csv('../input/schedule_5min_0612.csv').reset_index()
LGB = pd.read_csv('../input/demand_variation/schedule/alpha_7_demand_1500/7_1.csv').reset_index()
WDHL = pd.read_csv('../input/demand_variation/schedule/alpha_7_demand_500/6_1.csv').reset_index()
ELSG = pd.read_csv('../input/demand_variation/schedule/alpha_7_demand_2500/5_1.csv').reset_index()
UVS = pd.read_csv('../input/demand_variation/schedule/alpha_7_demand_500/3_1.csv').reset_index()


DTLA['schedule'] = np.ceil(DTLA['schedule']/5)
LGB['schedule'] = np.ceil(LGB['schedule']/5)
WDHL['schedule'] = np.ceil(WDHL['schedule']/5)
UVS['schedule'] = np.ceil(UVS['schedule']/5)

DTLA['origin'] = DTLA['od'].apply(lambda x: 0 if x.split('_')[0] == 'LAX' else 1)
LGB['origin'] = LGB['od'].apply(lambda x: 0 if x.split('_')[0] == 'LAX' else 2)
WDHL['origin'] = WDHL['od'].apply(lambda x: 0 if x.split('_')[0] == 'LAX' else 3)
UVS['origin'] = UVS['od'].apply(lambda x: 0 if x.split('_')[0] == 'LAX' else 4)

DTLA['destination'] = DTLA['od'].apply(lambda x: 0 if x.split('_')[1] == 'LAX' else 1)
LGB['destination'] = LGB['od'].apply(lambda x: 0 if x.split('_')[1] == 'LAX' else 2)
WDHL['destination'] = WDHL['od'].apply(lambda x: 0 if x.split('_')[1] == 'LAX' else 3)
UVS['destination'] = UVS['od'].apply(lambda x: 0 if x.split('_')[1] == 'LAX' else 4)

vertiports = ['LAX', 'DTLA', 'LGB', 'WDHL', 'UVS']

flight_time = np.array([[0,2,3,4,4],
                        [2,0,5,6,6],
                        [3,5,0,7,7],
                        [4,6,7,0,8],
                        [4,6,7,8,0]])

# Units for energy consumption is levels of %SoC
energy_consumption = np.array([[0,10,15,20,25],
                               [10,0,25,30,35],
                               [15,25,0,35,35],
                               [20,30,35,0,40],
                               [25,35,35,40,0]]) / 2.5
energy_consumption = energy_consumption.astype(int)

od_matrix = np.array([[0,1,1,1,1],
                      [1,0,1,1,1],
                      [1,1,0,1,1],
                      [1,1,1,0,1],
                      [1,1,1,1,0]])


network2 = ChargingNetwork(vertiports, flight_time, energy_consumption, od_matrix)
nodes, supply, edges, cost, c = network2.populate_network()

def obtain_demand_dict(df, existing_dict=None):
    if existing_dict is not None:
        demand_dict = existing_dict
    else:
        demand_dict = {}
        
    df_grouped = df.groupby(['origin', 'destination', 'schedule']).size().reset_index(name='count')
    for i, row in df_grouped.iterrows():
        demand_dict[(int(row['origin']), int(row['destination']), int(row['schedule']))] = int(row['count'])
    return demand_dict

od_key = {'LAX':0, 'DTLA':1, 'LGB':2, 'WDHL':3, 'UVS':4}
flight_demand = obtain_demand_dict(DTLA)
flight_demand = obtain_demand_dict(LGB, flight_demand)
flight_demand = obtain_demand_dict(WDHL, flight_demand)
flight_demand = obtain_demand_dict(UVS, flight_demand)

def get_flight_edges(i, model_param, flight_time=flight_time, energy_consumption=energy_consumption):
    K = 32
    v1, v2, t = i
    flight_time_ij = flight_time[v1, v2]
    soc_level_drop = int(energy_consumption[v1, v2])

    volume = quicksum(model_param[(v1, t, k), (v2, t+flight_time_ij, k-soc_level_drop)] for k in range(soc_level_drop, K+1))
    
    return volume
    

def solve_lr(m, c, flow, edges, flight_demand, lambda_s, step_size):

    # lambda_s_i_1 = lambda_s

    primal_objective = quicksum(c[i,j] * flow[i,j] for i,j in edges)
    lagragian_dual_objective = quicksum(lambda_s[i] * (flight_demand[i] - get_flight_edges(i, flow)) for i in flight_demand.keys())
    
    m.setObjective(primal_objective + lagragian_dual_objective, GRB.MINIMIZE)
    m.optimize()

    subgradients = {i: flight_demand[i] - get_flight_edges(i, flow).getValue() for i in flight_demand.keys()}
    solution = m.getAttr('x', flow)
    flow_sum = sum(solution[i,j] for (i,j) in edges if i == 'Source' and j == 'Sink')

    primal_ofv = primal_objective.getValue()
    dual_ofv = lagragian_dual_objective.getValue()
    fleetsize = primal_ofv + dual_ofv

    for i in flight_demand.keys():
        lambda_s[i] = max(0, lambda_s[i] + step_size * subgradients[i])

    # for i in flight_demand.keys():
    #     if (lambda_s[i] == 0) & (lambda_s_i_1[i] > 0):
    #         lambda_s[i] = lambda_s_i_1[i]

    # Return primal and dual objective values, fleetsize, and flow sum, and updated lambda_s
    return primal_ofv, dual_ofv, fleetsize, flow_sum, lambda_s
    





def main():
    with open('../output/star_network/LR/dual_prices.pkl', 'rb') as f:
        lambda_s = pickle.load(f)

    start_time = time()
    lrm = gp.read('../output/star_network/LR/NETWORK_Gurobipy.mps')
    flow_vars = lrm.getVars()
    flow = {edges[idx]: flow_vars[idx] for idx in range(len(flow_vars))}
    lrm.setParam('OutputFlag', 0)


    max_iter = 10000
    fs = []
    for iteration in range(max_iter):
        step_size = 0.0001
        lrm = gp.read('../output/star_network/LR/NETWORK_Gurobipy.mps')
        flow_vars = lrm.getVars()
        flow = {edges[idx]: flow_vars[idx] for idx in range(len(flow_vars))}
        lrm.setParam('OutputFlag', 0)
        
        primal_ofv, dual_ofv, fleetsize, flow_sum, lambda_s = solve_lr(lrm, c, flow, edges, flight_demand, lambda_s, step_size)

        print('------------Iteration {}------------'.format(iteration))
        print('Primal Objective: {}'.format(primal_ofv))
        print('Dual Objective: {}'.format(dual_ofv))
        print('Fleetsize: {}'.format(fleetsize))
        print('Flow Sum: {}'.format(flow_sum))

        fs.append(fleetsize)
        
        if (iteration % 50 == 0) & (iteration > 0):
            print('Time Elapsed: {}'.format(time() - start_time))
            fig, ax = plt.subplots(dpi=200, figsize=(6,4))
            sns.lineplot(fs, ax=ax)
            ax.set(xlim=(0, len(fs)), ylim=(0, 60), 
                xlabel='Iteration', ylabel='Fleetsize', title='Subgradient Update');
            ax.grid(True, alpha=0.4, linestyle='--', which='both')
            
            plt.savefig('../output/star_network/LR/fleetsize.png', bbox_inches='tight')

            with open('../output/star_network/LR/lagragian_multipliers.pkl', 'wb') as f:
                pickle.dump((lambda_s, fs), f)
            with open('../output/star_network/LR/fleetsize.pkl', 'wb') as f:
                pickle.dump(fleetsize, f)

if __name__ == '__main__':
    main()
