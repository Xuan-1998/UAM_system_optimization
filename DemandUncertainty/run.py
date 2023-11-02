import importlib
import auto_regressive_demand
importlib.reload(auto_regressive_demand)
import auto_regressive_demand as ard
from op import number_aircrafts_lp_v2
import os
import numpy as np

# if __name__ == '__main__':
#     schedule = ard.UAM_Schedule('/DemandUncertainty/LAX_ind.csv', '/DemandUncertainty/T_F41SCHEDULE_B43.csv')
#     alpha = 0.7

#     for month in range(1, 13):
#         for day in range(1, 32):
#             if (month == 2 and day > 28) or ((month in [4, 6, 9, 11]) and day > 30):
#                 continue  # Skip invalid dates
            
#             # Here, you can perform your tasks for each valid date
#             print(f"Year: 2019, Month: {month}, Day: {day}")
#             one_day = schedule.get_one_day(month, day, alpha)
#             one_day.to_csv(os.getcwd() + f'/output/demand_variation/schedule/{month}_{day}_{int(alpha*10)}.csv', index=False)
#             number_aircrafts_lp(schedule=one_day, schedule_time_step=288, output_path=f'/output/demand_variation/results/{month}_{day}_{int(alpha*10)}')




if __name__ == '__main__':

    directory_path = "output/demand_variation/results"  # Replace with the path to your directory
    file_list = os.listdir(directory_path)

    all_files = []
    for filename in file_list:
        if filename.endswith("_7_fleetsize.txt"):
            all_files.append(filename)
    file_names = np.empty(shape=(0, 2))
    for i in all_files:
        file_names = np.vstack((file_names, np.array(i.split('_')[:2])))

    file_names = file_names.astype(int)

    schedule = ard.UAM_Schedule('DemandUncertainty/LAX_ind.csv', 'DemandUncertainty/T_F41SCHEDULE_B43.csv')
    alpha = 0.7

    for month in range(1, 13):
        for day in range(1, 32):
            if (month == 2 and day > 28) or ((month in [4, 6, 9, 11]) and day > 30):
                continue  # Skip invalid dates

            if (np.all(file_names == np.array([month,day]), axis=1).any() == False):
                print(f"Year: 2019, Month: {month}, Day: {day}")
                one_day = schedule.get_one_day(month, day, alpha)
                one_day.to_csv(f'output/demand_variation/schedule/{month}_{day}_{int(alpha*10)}.csv', index=False)
                number_aircrafts_lp_v2(schedule=one_day, schedule_time_step=288, output_path=f'output/demand_variation/results/{month}_{day}_{int(alpha*10)}')