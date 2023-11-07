from UAM_Schedule import UAM_Schedule
from spill_op import spill_op
import os
import multiprocessing
import argparse
import numpy as np
import pandas as pd
import re



def optimize(month, day, fleet_size, alpha):

    flight_schedule = pd.read_csv(f'output/demand_variation/schedule/alpha_{int(alpha*10)}/{month}_{day}.csv')
    occupancy = pd.read_csv(f'output/demand_variation/schedule/alpha_{int(alpha*10)}/num_pax_{month}_{day}.csv')
    print(f"Year: 2019, Month: {month}, Day: {day}, Fleet size: {fleet_size}")
    spill_op(flight_schedule, occupancy, schedule_time_step=288, fleet_size=fleet_size, output_path=f'output/demand_variation/spill_op_result/alpha_{int(alpha*10)}/{month}_{day}_{fleet_size}')
        


        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", "-a", type=float, default=0.7)
    parser.add_argument("--n_cores", "-n", type=int, default=32)
    parser.add_argument("--selected", "-s", type=bool, default=False)
    parser.add_argument("--min_fleetsize", "-fsmin", type=int, default=15)
    parser.add_argument("--max_fleetsize", "-fsmax", type=int, default=18)
    args = parser.parse_args()

    alpha = args.alpha
    num_processes = args.n_cores  # Number of CPU cores

    if args.selected:
        directory_path = f"output/demand_variation/spill_op_result/alpha_{int(alpha*10)}"  # Replace with the path to your directory
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
            
                with open(f'output/demand_variation/fleet_op_result/alpha_7/{month}_{day}_fleetsize.txt'.format(month, day), 'r') as f:
                    fleetsize = int(float(re.search(r'\d+(\.\d+)?', f.readline())[0]))
                for i in range(args.min_fleetsize, args.max_fleetsize+1):
                    if fleetsize > i:
                        if (np.all(file_names == np.array([month,day,i]), axis=1).any() == False):
                            valid_dates.append((month, day, i, alpha))
    else:
        valid_dates = []
        for month in range(1, 13):
            for day in range(1, 32):
                if (month == 2 and day > 28) or ((month in [4, 6, 9, 11]) and day > 30):
                    continue  # Skip invalid dates
                else:
                    with open(f'output/demand_variation/fleet_op_result/alpha_7/{month}_{day}_fleetsize.txt'.format(month, day), 'r') as f:
                        fleetsize = int(float(re.search(r'\d+(\.\d+)?', f.readline())[0]))
                    
                    for i in range(args.min_fleetsize, args.max_fleetsize+1):
                        if fleetsize > i:
                            valid_dates.append((month, day, i, alpha))

    with multiprocessing.Pool(num_processes) as p:
        p.starmap(optimize, valid_dates)
        p.wait()




    


