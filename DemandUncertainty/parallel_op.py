from UAM_Schedule import UAM_Schedule
from op import FleetSizeOptimizer
import multiprocessing
import argparse
import numpy as np
import os



def optimize(month, day, alpha, demand):
    # Create a new instance of UAM_Schedule within each child process
    schedule = UAM_Schedule('DemandUncertainty/LAX_ind.csv', 'DemandUncertainty/T_F41SCHEDULE_B43.csv')
    print(f"Year: 2019, Month: {month}, Day: {day}")
    one_day, pax_arrival, num_pax_per_flight = schedule.get_one_day(month, day, alpha, directional_demand=demand)
    one_day.to_csv(f'input/demand_variation/schedule/alpha_{int(alpha*10)}_demand_{demand}/{month}_{day}.csv', index=False)
    num_pax_per_flight.to_csv(f'input/demand_variation/schedule/alpha_{int(alpha*10)}_demand_{demand}/num_pax_{month}_{day}.csv', index=False)
    pax_arrival.to_csv(f'input/demand_variation/passenger_arrival/alpha_{int(alpha*10)}_demand_{demand}/{month}_{day}.csv', index=False)

    model = FleetSizeOptimizer(flight_time=np.array([[0, 10], [10, 0]]), energy_consumption=np.array([[0, 10], [10, 0]]), 
                               schedule=f'demand_variation/schedule/alpha_{int(alpha*10)}_demand_{demand}/{month}_{day}.csv')
    model.optimize(output_path=f'demand_variation/fleet_op_result/alpha_{int(alpha*10)}_demand_{demand}/{month}_{day}', verbose=False, optimality_gap=0.1)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", "-a", type=float, default=0.7)
    parser.add_argument("--demand", "-d", type=int, default=2500)
    parser.add_argument("--n_cores", "-n", type=int, default=12)
    parser.add_argument("--selected", "-s", type=bool, default=False)
    args = parser.parse_args()


    alpha = args.alpha
    num_processes = args.n_cores  # Number of CPU cores
    demand = args.demand

    if args.selected:
        directory_path = f"output/demand_variation/fleet_op_result/alpha_{int(alpha*10)}_demand_{demand}"  # Replace with the path to your directory
        file_list = os.listdir(directory_path)
        all_files = []
        for filename in file_list:
            if filename.endswith("_fleetsize.txt"):
                all_files.append(filename)
        file_names = np.empty(shape=(0, 2))
        for i in all_files:
            file_names = np.vstack((file_names, np.array(i.split('_')[:2])))

        file_names = file_names.astype(int)
        valid_dates = []
        for month in range(1, 13):
            for day in range(1, 32):
                if (month == 2 and day > 28) or ((month in [4, 6, 9, 11]) and day > 30):
                    continue  # Skip invalid dates
                if (np.all(file_names == np.array([month,day]), axis=1).any() == False):
                    valid_dates.append((month, day, alpha, demand))
    else:
        valid_dates = [(month, day, alpha, demand) for month in range(1, 13) for day in range(1, 32) if (month == 2 and day <= 28) or (month not in [4, 6, 9, 11] or day <= 30)]
    print(f'Expecting {len(valid_dates)} runs')

    with multiprocessing.Pool(num_processes) as p:
        p.starmap(optimize, valid_dates)




    


