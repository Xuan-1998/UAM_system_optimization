import pandas as pd
import numpy as np
from utils import generate_uam_schedule
import os

class UAM_Schedule:
    def __init__(self, path_schedule, path_seatcap):
        self.lax_flight = pd.read_csv(path_schedule)
        self.lax_flight['T_OAG_S_DE'] = pd.to_datetime(self.lax_flight['T_OAG_S_DE'])
        self.lax_flight['T_OOOI_ARR'] = pd.to_datetime(self.lax_flight['T_OOOI_ARR'])
        self.seatcap = pd.read_csv(path_seatcap)
    
    def get_one_day(self, month, day, auto_regressive_alpha=None):
        month = month+201900

        lax_flight_arr = self.lax_flight[(self.lax_flight['ON_YYYYMM'] == month) & (self.lax_flight['ON_DAY'] == day) & (self.lax_flight['USER_CLASS'] == 'C') & (self.lax_flight['ARR_LOCID'] == ' LAX')]
        lax_flight_arr['time'] = lax_flight_arr['T_OOOI_ARR'].dt.hour * 60 + lax_flight_arr['T_OOOI_ARR'].dt.minute
        lax_flight_arr = lax_flight_arr.sort_values('time')
        lax_flight_dep = self.lax_flight[(self.lax_flight['ON_YYYYMM'] == month) & (self.lax_flight['ON_DAY'] == day) & (self.lax_flight['USER_CLASS'] == 'C') & (self.lax_flight['DEP_LOCID'] == ' LAX')]
        lax_flight_dep['time'] = lax_flight_dep['T_OAG_S_DE'].dt.hour * 60 + lax_flight_dep['T_OAG_S_DE'].dt.minute
        lax_flight_dep = lax_flight_dep.sort_values('time')

        schedule, pax_arrival_times = generate_uam_schedule(lax_flight_arr, lax_flight_dep, seatcap=self.seatcap, auto_regressive_alpha=auto_regressive_alpha, directional_demand=1500, max_waiting_time=5)
        
        return schedule, pax_arrival_times





        



