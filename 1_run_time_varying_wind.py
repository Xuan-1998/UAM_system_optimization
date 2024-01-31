from model.op import FleetSizeOptimizer
import multiprocessing
import argparse
import numpy as np
import os
import pickle

def optimize(run_id, flight_time, energy_consumption):
    # Create a new instance of UAM_Schedule within each child process
    print(f"Running {run_id}")
    model = FleetSizeOptimizer(flight_time=flight_time, energy_consumption=energy_consumption, 
                               schedule=f'ICRAT_wind/schedule_1500pax_5min_0125.csv')
    model.optimize(output_path=f'ICRAT_wind/fleet_op_result/{run_id}', verbose=False)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_cores", "-n", type=int, default=8)
    parser.add_argument("--selected", "-s", type=bool, default=True)
    args = parser.parse_args()
    num_processes = args.n_cores  # Number of CPU cores

    with open('input/wind/wind_params_updated.pkl', 'rb') as f:
        params = pickle.load(f)

    if args.selected:
        file_list = os.listdir('output/ICRAT_wind/fleet_op_result')
        all_files = []
        for filename in file_list:
            if filename.endswith('_fleetsize.txt'):
                all_files.append(filename)
        file_names = np.empty(shape=(1, 4))
        for i in all_files:
            file_names = np.vstack((file_names, np.array(i.split('_')[:4])))
        file_names = file_names[1:,[1,3]].astype(int)

        valid_runs = []
        for i in range(20, 160, 10):
            for j in [500, 600, 700, 800, 900, 950, 990, 995]:
                if (np.all(file_names == np.array([i,j]), axis=1).any() == False):
                    valid_runs.append((f'dist_{i}_per_{j}', params[f'dist_{i}_per_{j}']['flight_time'], params[f'dist_{i}_per_{j}']['energy_consumption']))

    else:
        valid_runs = [(run_id, params[run_id]['flight_time'], params[run_id]['energy_consumption']) for run_id in params.keys()]
 
    print(f'Expecting {len(valid_runs)} runs')
    with multiprocessing.Pool(num_processes) as p:
        p.starmap(optimize, valid_runs)




    


