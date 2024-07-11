from model.op import FleetSizeOptimizer
import multiprocessing
import argparse
import numpy as np
import os
import pickle

def optimize(run_id, flight_time, energy_consumption, optimality_gap):
    # Create a new instance of UAM_Schedule within each child process
    print(f"Running {run_id}")
    model = FleetSizeOptimizer(flight_time=flight_time, energy_consumption=energy_consumption, 
                               schedule=f'icrat_wind/demand/schedule_500pax_5min_0206.csv')
    model.optimize(output_path=f'icrat_wind/fleet_op_result/{run_id}', verbose=False, optimality_gap=optimality_gap)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_cores", "-n", type=int, default=8)
    parser.add_argument("--selected", "-s", type=bool, default=True)
    parser.add_argument("--optimality_gap", "-o", type=float, default=0.05)
    args = parser.parse_args()
    num_processes = args.n_cores  # Number of CPU cores
    optimality_gap = args.optimality_gap

    with open('input/icrat_wind/parsed_aircraft_params_zero_wind.pkl', 'rb') as f:
        params = pickle.load(f)

    if args.selected:
        file_list = os.listdir('output/icrat_wind/fleet_op_result')
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
            if (np.all(file_names == np.array([i,99]), axis=1).any() == False):
                valid_runs.append((f'dist_{i}_cluster_99', params[f'dist_{i}_cluster_99']['flight_time'], params[f'dist_{i}_cluster_99']['energy_consumption'], optimality_gap))

    else:
        valid_runs = [(run_id, params[run_id]['flight_time'], params[run_id]['energy_consumption'], optimality_gap) for run_id in params.keys()]
 
    print(f'Expecting {len(valid_runs)} runs')
    with multiprocessing.Pool(num_processes) as p:
        p.starmap(optimize, valid_runs)




    


