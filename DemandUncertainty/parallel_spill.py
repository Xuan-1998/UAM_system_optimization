# from UAM_Schedule import UAM_Schedule
from model.IP import FleetSizeOptimizer
import os
import multiprocessing
import argparse
import numpy as np
import pandas as pd
import re


def optimize(month, day, fleet_size, alpha, demand, optimality_gap):
    try:
        occupancy = pd.read_csv(f'input/demand_variation/schedule/alpha_{int(alpha*10)}_demand_{demand}/num_pax_{month}_{day}.csv')
        print(f"Year: 2019, Month: {month}, Day: {day}, Fleet size: {fleet_size}")

        model = FleetSizeOptimizer(
            flight_time=np.array([[0, 10], [10, 0]]),
            energy_consumption=np.array([[0, 10], [10, 0]]),
            schedule=f'input/demand_variation/schedule/alpha_{int(alpha*10)}_demand_{demand}/{month}_{day}.csv'
        )
        model.optimize(
            output_path=f'input/demand_variation/spill_op_result/alpha_{int(alpha*10)}_demand_{demand}/{month}_{day}_{fleet_size}',
            occupancy=occupancy,
            fleet_size=fleet_size,
            verbose=False,
            optimality_gap=optimality_gap,
            spill_optimization=True,
            seat_capacity=4
        )
    except Exception as e:
        print(f"An error occurred for Month: {month}, Day: {day}, Fleet size: {fleet_size}")
        print(f"Error: {e}")


        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", "-a", type=float, default=0.7)
    parser.add_argument("--n_cores", "-n", type=int, default=24)
    parser.add_argument("--selected", "-s", type=bool, default=True)
    parser.add_argument("--demand", "-d", type=int, default=500)
    parser.add_argument("--optimality_gap", "-o", type=float, default=0.05)
    parser.add_argument("--min_fleetsize", "-fsmin", type=int, default=25)
    parser.add_argument("--max_fleetsize", "-fsmax", type=int, default=40)
    args = parser.parse_args()

    alpha = args.alpha
    demand = args.demand
    num_processes = args.n_cores  # Number of CPU cores
    optimality_gap = args.optimality_gap

    if args.selected:
        directory_path = f"output/demand_variation/spill_op_result/alpha_{int(alpha*10)}_demand_{demand}"  # Replace with the path to your directory
        file_list = os.listdir(directory_path)
        all_files = []
        for filename in file_list:
            if filename.endswith("_total_spill.txt"):
                all_files.append(filename)
        file_names = np.empty(shape=(0, 3))
        for i in all_files:
            file_names = np.vstack((file_names, np.array(i.split('_')[:3])))

        file_names = file_names.astype(int)
        valid_dates = []
        for month in range(1, 13):
            for day in range(1, 32):
                if (month == 2 and day > 28) or ((month in [4, 6, 9, 11]) and day > 30):
                    continue  # Skip invalid dates
                
                try:
                    with open(f'output/demand_variation/fleet_op_result/alpha_{int(alpha*10)}_demand_{demand}/{month}_{day}_fleetsize.txt'.format(month, day), 'r') as f:
                        fleetsize = int(float(re.search(r'\d+(\.\d+)?', f.readline())[0]))
                    for i in range(args.min_fleetsize, args.max_fleetsize+1):
                        if fleetsize > i:
                            if (np.all(file_names == np.array([month,day,i]), axis=1).any() == False):
                                valid_dates.append((month, day, i, alpha, demand, optimality_gap))
                except FileNotFoundError:
                    continue

    else:
        valid_dates = []
        for month in range(1, 13):
            for day in range(1, 32):
                if (month == 2 and day > 28) or ((month in [4, 6, 9, 11]) and day > 30):
                    continue  # Skip invalid dates
                else:
                    with open(f'output/demand_variation/fleet_op_result/alpha_{int(alpha*10)}_demand_{demand}/{month}_{day}_fleetsize.txt'.format(month, day), 'r') as f:
                        fleetsize = int(float(re.search(r'\d+(\.\d+)?', f.readline())[0]))
                    
                    for i in range(args.min_fleetsize, args.max_fleetsize+1):
                        if fleetsize > i:
                            valid_dates.append((month, day, i, alpha, demand, optimality_gap))

    print(f'Expecting {len(valid_dates)} runs')
    
    with multiprocessing.Pool(num_processes) as p:
        p.starmap(optimize, valid_dates)




    


