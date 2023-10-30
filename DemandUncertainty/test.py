import importlib
import auto_regressive_demand
importlib.reload(auto_regressive_demand)
import auto_regressive_demand as ard
from op import number_aircrafts_lp
import tqdm as tqdm

if __name__ == '__main__':
    schedule = ard.UAM_Schedule('/DemandUncertainty/LAX_ind.csv', '/DemandUncertainty/T_F41SCHEDULE_B43.csv')
    alpha = 0.6

    for month in tqdm(range(1, 13)):
        for day in range(1, 32):
            if (month == 2 and day > 28) or ((month in [4, 6, 9, 11]) and day > 30):
                continue  # Skip invalid dates
            
            # Here, you can perform your tasks for each valid date
            print(f"Year: 2019, Month: {month}, Day: {day}")
            one_day = schedule.get_one_day(month, day, alpha)
            number_aircrafts_lp(schedule=one_day, schedule_time_step=288, output_path=f'/output/demand_variation/{month}_{day}_{alpha*10}')




