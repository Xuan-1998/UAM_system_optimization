from model.op import FleetSizeOptimizer
import multiprocessing
import argparse
import numpy as np
import os
import pickle

def optimize(run_id, params):
    # Create a new instance of UAM_Schedule within each child process
    print(f"Running {run_id}")
    model = FleetSizeOptimizer(flight_time=params[run_id]['flight_time'], energy_consumption=params[run_id]['energy_consumption'], 
                               schedule=f'ICRAT_wind/schedule_1500pax_5min_0125.csv')
    model.optimize(output_path=f'ICRAT_wind/fleet_op_result/{run_id}', verbose=False, optimality_gap=0.05)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_cores", "-n", type=int, default=8)
    args = parser.parse_args()
    num_processes = args.n_cores  # Number of CPU cores

    with open('input/wind/wind_params.pkl', 'rb') as f:
        params = pickle.load(f)

    valid_runs = [(run_id, params) for run_id in params.keys()]
 
    print(f'Expecting {len(valid_runs)} runs')

    with multiprocessing.Pool(num_processes) as p:
        p.starmap(optimize, valid_runs)




    


