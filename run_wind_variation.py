import os
import multiprocessing.dummy as multiprocessing
import argparse
import model.op
from model.op import FleetSizeOptimizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3


def optimize(tau, kappa, key):
    distance_key, wind_direction_key, wind_magnitude_key = key
    # tau = data_dict[(distance_key, wind_direction_key, wind_magnitude_key)]['flight_time']
    # kappa = data_dict[(distance_key, wind_direction_key, wind_magnitude_key)]['energy_consumption']

    op = FleetSizeOptimizer(flight_time=tau, energy_consumption=kappa, schedule='schedule_5min_0612.csv')
    op.optimize(f'wind_variation/{distance_key}_{wind_direction_key}_{wind_magnitude_key}.txt', 
                charging_station=[True, True],
                verbose=False)
    print(f'Finished {distance_key}_{wind_direction_key}_{wind_magnitude_key}')
    # op.parse_result(f'wind_variation/{distance_key}_{wind_direction_key}_{wind_magnitude_key}.txt')
    # op.calculate_aircraft_states()

    # dist = np.array([[0, distance_key], [distance_key, 0]])
    # fleet_size, number_of_pads, pads_by_vertiport, num_flight, energy_consumption, am, ram = op.get_summary_statistics(dist, return_summary=True)

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_cores", "-n", type=int, default=32)
    parser.add_argument("--selected", "-s", type=bool, default=False)
    args = parser.parse_args()

    num_processes = args.n_cores  # Number of CPU cores

    directory_path = f"output/wind_variation"  # Replace with the path to your directory
    file_list = os.listdir(directory_path)
    file_names = np.empty(shape=(0, 3))
    for i in file_list:
        file_names = np.vstack((file_names, np.array(i.split('_')[:3])))
    file_names = file_names.astype(int)

    sql_table = pd.DataFrame()
    for i in range(20, 70, 10):
        conn = sqlite3.connect(f'input/wind/energy_and_flight_time_{i}_mile_route.sqlite')
        query = """
        SELECT flight_direction, energy_consumption, flight_time, wind_direction_degrees, wind_magnitude_mph
        FROM flight_metrics
        """
        df = pd.read_sql_query(query, conn)
        df['distance'] = i

        sql_table = pd.concat([sql_table, df])  
    sql_table = sql_table.reset_index(drop=True)
    data_dict = {}


    for i in range(0, len(sql_table), 2):
        k1 = sql_table['energy_consumption'][i]
        k2 = sql_table['energy_consumption'][i+1]
        kappa = np.array([[0, k1],[k2, 0]])/160*100

        t1 = sql_table['flight_time'][i]
        t2 = sql_table['flight_time'][i+1]
        tau = np.array([[0, t1],[t2, 0]])+5

        distance = sql_table['distance'][i]
        wind_direction_degrees = sql_table['wind_direction_degrees'][i]
        wind_magnitude_mph = sql_table['wind_magnitude_mph'][i]

        key = (distance, wind_direction_degrees, wind_magnitude_mph)
        data_dict[key] = {'flight_time': tau, 'energy_consumption': kappa}

    inputs = []
    for i in range(20, 70, 10):
        for j in range(0, 270, 90):
            for k in range(10, 50, 10):
                if args.selected:
                    if (i in file_names[:, 0]) and (j in file_names[:, 1]) and (k in file_names[:, 2]):
                        continue
                    else:
                        inputs.append((data_dict[(i, j, k)]['flight_time'], data_dict[(i, j, k)]['energy_consumption'], (i,j,k)))
        if (0 in file_names[:, 0]) and (0 in file_names[:, 1]) and (i in file_names[:, 2]):
            continue
        else:
            inputs.append((data_dict[(i, 0, 0)]['flight_time'], data_dict[(i, 0, 0)]['energy_consumption'], (i,0,0)))


    with multiprocessing.Pool(num_processes) as p:
        p.starmap(optimize, inputs)




    


