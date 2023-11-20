from UAM_Schedule import UAM_Schedule
from op import number_aircrafts_lp_v2
import os
import multiprocessing
import argparse
import numpy as np



def optimize(month, day, alpha):
    # Create a new instance of UAM_Schedule within each child process
    schedule = UAM_Schedule('DemandUncertainty/LAX_ind.csv', 'DemandUncertainty/T_F41SCHEDULE_B43.csv')
    print(f"Year: 2019, Month: {month}, Day: {day}")
    one_day, pax_arrival, num_pax_per_flight = schedule.get_one_day(month, day, alpha)
    one_day.to_csv(f'output/demand_variation/schedule/alpha_{int(alpha*10)}/{month}_{day}.csv', index=False)
    num_pax_per_flight.to_csv(f'output/demand_variation/schedule/alpha_{int(alpha*10)}/num_pax_{month}_{day}.csv', index=False)
    pax_arrival.to_csv(f'output/demand_variation/passenger_arrival/alpha_{int(alpha*10)}/{month}_{day}.csv', index=False)

    number_aircrafts_lp_v2(schedule=one_day, schedule_time_step=288, output_path=f'output/demand_variation/fleet_op_result/alpha_{int(alpha*10)}/{month}_{day}')
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", "-a", type=float, default=0.7)
    parser.add_argument("--n_cores", "-n", type=int, default=32)
    parser.add_argument("--selected", "-s", type=bool, default=False)
    args = parser.parse_args()


    alpha = args.alpha
    num_processes = args.n_cores  # Number of CPU cores

    if args.selected:
        directory_path = f"output/demand_variation/fleet_op_result/alpha_{int(alpha*10)}"  # Replace with the path to your directory
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
                    valid_dates.append((month, day, alpha))
    else:
        valid_dates = [(month, day, alpha) for month in range(1, 13) for day in range(1, 32) if (month == 2 and day <= 28) or (month not in [4, 6, 9, 11] or day <= 30)]


    with multiprocessing.Pool(num_processes) as p:
        p.starmap(optimize, valid_dates)




    


