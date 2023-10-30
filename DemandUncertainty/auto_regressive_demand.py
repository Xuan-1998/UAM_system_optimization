import pandas as pd
import numpy as np
from utils import generate_uam_schedule, pois_generate

import os
class UAM_Schedule:
    def __init__(self, path_schedule, path_seatcap):
        self.lax_flight = pd.read_csv(os.getcwd()+path_schedule)
        self.lax_flight['T_OAG_S_DE'] = pd.to_datetime(self.lax_flight['T_OAG_S_DE'])
        self.lax_flight['T_OOOI_ARR'] = pd.to_datetime(self.lax_flight['T_OOOI_ARR'])
        self.seatcap = pd.read_csv(os.getcwd()+path_seatcap)
    
    def get_one_day(self, month, day, auto_regressive=None):
        month = month+201900

        lax_flight_arr = self.lax_flight[(self.lax_flight['ON_YYYYMM'] == month) & (self.lax_flight['ON_DAY'] == day) & (self.lax_flight['USER_CLASS'] == 'C') & (self.lax_flight['ARR_LOCID'] == ' LAX')]
        lax_flight_arr['time'] = lax_flight_arr['T_OOOI_ARR'].dt.hour * 60 + lax_flight_arr['T_OOOI_ARR'].dt.minute
        lax_flight_arr = lax_flight_arr.sort_values('time')
        lax_flight_dep = self.lax_flight[(self.lax_flight['ON_YYYYMM'] == month) & (self.lax_flight['ON_DAY'] == day) & (self.lax_flight['USER_CLASS'] == 'C') & (self.lax_flight['DEP_LOCID'] == ' LAX')]
        lax_flight_dep['time'] = lax_flight_dep['T_OAG_S_DE'].dt.hour * 60 + lax_flight_dep['T_OAG_S_DE'].dt.minute
        lax_flight_dep = lax_flight_dep.sort_values('time')
        schedule = generate_uam_schedule(lax_flight_arr, lax_flight_dep, seatcap=self.seatcap, directional_demand=1500, max_waiting_time=5)

        if auto_regressive==None:
            return schedule
        
        else:
            schedule['hour'] = schedule['schedule'] // 60
            hourly_rate = schedule.groupby(['hour', 'od']).count().reset_index()
            lax_dtla_rate = hourly_rate[hourly_rate['od'] == 'LAX_DTLA']['schedule'].to_numpy()
            dtla_lax_rate = hourly_rate[hourly_rate['od'] == 'DTLA_LAX']['schedule'].to_numpy()

            lax_dtla_rate = pois_generate(lax_dtla_rate, alpha=auto_regressive)
            dtla_lax_rate = pois_generate(dtla_lax_rate, alpha=auto_regressive)

            dtla_lax_schedule = np.empty((0,))
            for idx, value in enumerate(dtla_lax_rate):
                dtla_lax_schedule = np.concatenate([dtla_lax_schedule, np.random.uniform(idx*60, idx*60+60, size=value)])

            lax_dtla_schedule = np.empty((0,))
            for idx, value in enumerate(lax_dtla_rate):
                lax_dtla_schedule = np.concatenate([lax_dtla_schedule, np.random.uniform(idx*60, idx*60+60, size=value)])

            schedule = pd.DataFrame({'schedule': np.concatenate([lax_dtla_schedule, dtla_lax_schedule], axis=0),
                                     'od': np.concatenate([np.repeat('LAX_DTLA', len(lax_dtla_schedule)), np.repeat('DTLA_LAX', len(dtla_lax_schedule))], axis=0)})
            return schedule





        



