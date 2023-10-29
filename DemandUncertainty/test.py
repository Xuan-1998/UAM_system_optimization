import importlib
import auto_regressive_demand
importlib.reload(auto_regressive_demand)
import auto_regressive_demand as ard
from op import number_aircrafts_lp



schedule = ard.UAM_Schedule('/DemandUncertainty/LAX_ind.csv', '/DemandUncertainty/T_F41SCHEDULE_B43.csv')
month = 3
day = 2
alpha = 0.6
one_day = schedule.get_one_day(month, day, alpha)
number_aircrafts_lp(schedule=one_day, schedule_time_step=288, output_path='demand_variation/{month}+{day}+{alpha}')

