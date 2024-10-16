import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



class one_vertiport_system:
    def __init__(self, fleet_size, num_pax_flight_path, flight_schedule_path, print_log=False):
        self.print_log = print_log
        self.fleet_size = fleet_size
        num_pax_flight = pd.read_csv(num_pax_flight_path)
        flight_sche = pd.read_csv(flight_schedule_path)
        schedule = pd.concat([flight_sche, num_pax_flight['num_pax']], axis=1)
        schedule['schedule'] = np.ceil(schedule['schedule'] / 5)
        self.schedule = schedule

        self.vertiport = Vertiport([aircraft(aircraft_id="AA{:02d}".format(i), flight_time=2, charge_time=1, soc_change=10) for i in range(fleet_size)], print_log=print_log)
        self.arrival_curve = schedule.groupby('schedule').sum()['num_pax'].reset_index()

    def __update__(self, num_flight, num_pax):
        if self.print_log:
            print(f'------------------------------------------timestep------------------------------------------')

        num_flight_dispatched, departed_aircraft = self.vertiport.update(num_flight, one_vertiport_system=True)

        if len(departed_aircraft)*2 >= num_flight:
            pax_served = num_pax.sum()
        else:
            # pax_served = num_pax[:num_flight-len(departed_aircraft)*2].sum()
            pax_served = num_pax[:len(departed_aircraft)].sum()
        if self.print_log:
            print(pax_served, num_pax)
        return pax_served


    def logger(self, output_spill=False):
        departure_curve = []
        for i in range(1, 289):
            selected_flight = self.schedule[self.schedule['schedule'] == i]
            num_flights = selected_flight.shape[0]
            num_pax = selected_flight['num_pax'].to_numpy()
            num_pax = np.sort(num_pax)[::-1]
            total_pax = self.__update__(num_flights, num_pax)
            departure_curve.append(total_pax)

        departure_curve = pd.DataFrame({'schedule': np.arange(1, 289), 'num_pax': departure_curve})
        merged = self.arrival_curve.merge(departure_curve, how='outer', on='schedule').fillna(0).sort_values('schedule').reset_index(drop=True)
        merged.columns = ['t', 'at', 'dt']
        merged['cum_at'] = merged['at'].cumsum()
        merged['cum_dt'] = merged['dt'].cumsum()
        self.merged = merged
        if output_spill:
            return merged['cum_at'].max() - merged['cum_dt'].max()




class system:
    def __init__(self, fleet_size, num_pax_flight_path, flight_schedule_path, print_log=False):
        self.print_log = print_log
        self.fleet_size = fleet_size
        num_pax_flight = pd.read_csv(num_pax_flight_path)
        flight_sche = pd.read_csv(flight_schedule_path)
        schedule = pd.concat([flight_sche, num_pax_flight['num_pax']], axis=1)
        schedule['schedule'] = np.ceil(schedule['schedule'] / 5)
        self.schedule = schedule
        self.vertiports = [Vertiport([aircraft(aircraft_id="AA{:02d}".format(i), flight_time=2, charge_time=1, soc_change=10) for i in range(fleet_size//2)], print_log=print_log), 
                           Vertiport([aircraft(aircraft_id="AA{:02d}".format(fleet_size-i), flight_time=2, charge_time=1, soc_change=10) for i in range(fleet_size-fleet_size//2)], print_log=print_log)]

        self.arrival_curve = schedule.groupby('schedule').sum()['num_pax'].reset_index()

    def __update__(self, num_flight, num_pax):
        if self.print_log:
            print(f'------------------------------------------timestep------------------------------------------')

        num_flight_v1, departed_aircraft_v1v2 = self.vertiports[0].update(num_flight[0])
        num_flight_v2, departed_aircraft_v2v1 = self.vertiports[1].update(num_flight[1])
        
        self.vertiports[0].aircrafts.extend(departed_aircraft_v2v1)
        self.vertiports[1].aircrafts.extend(departed_aircraft_v1v2)

        if len(departed_aircraft_v1v2) >= num_flight[0]:
            pax_served_at_v1 = num_pax[0].sum()
        else:
            pax_served_at_v1 = num_pax[0][:num_flight[0]-num_flight_v1].sum()

        if len(departed_aircraft_v2v1) >= num_flight[1]:
            pax_served_at_v2 = num_pax[1].sum()
        else:
            pax_served_at_v2 = num_pax[1][:num_flight[1]-num_flight_v2].sum()
        if self.print_log:
            print(pax_served_at_v1, num_pax[0], pax_served_at_v2, num_pax[1])

        return pax_served_at_v1+pax_served_at_v2

    def logger(self, output_spill=False):
        departure_curve = []
        for i in range(1, 289):
            selected_flight = self.schedule[self.schedule['schedule'] == i]
            num_v1 = selected_flight[selected_flight['od'] == 'LAX_DTLA'].shape[0]
            num_v2 = selected_flight[selected_flight['od'] == 'DTLA_LAX'].shape[0]

            num_pax_v1 = selected_flight[selected_flight['od'] == 'LAX_DTLA']['num_pax'].to_numpy()
            num_pax_v2 = selected_flight[selected_flight['od'] == 'DTLA_LAX']['num_pax'].to_numpy()

            total_pax = self.__update__([num_v1, num_v2], [num_pax_v1, num_pax_v2])
            departure_curve.append(total_pax)

        departure_curve = pd.DataFrame({'schedule': np.arange(1, 289), 'num_pax': departure_curve})
        merged = self.arrival_curve.merge(departure_curve, how='outer', on='schedule').fillna(0).sort_values('schedule').reset_index(drop=True)
        merged.columns = ['t', 'at', 'dt']
        merged['cum_at'] = merged['at'].cumsum()
        merged['cum_dt'] = merged['dt'].cumsum()
        self.merged = merged

        if output_spill:
            return merged['cum_at'].max() - merged['cum_dt'].max()

    def plot(self, ax):
        ax.plot(self.merged['t'], self.merged['cum_at'], label='Arrival', color='red')
        ax.plot(self.merged['t'], self.merged['cum_dt'], label='Departure', color='blue')
        ax.set(xlabel='Time', ylabel='Number of passengers', title='N-T diagram',
               xticks=np.arange(0, 289, 96), xticklabels=['12AM', '8AM', '4PM', '12AM'],
               xlim=[0, 288], ylim=[0, 3500])
        ax.text(160, 500, f'Fleet Size: {self.fleet_size}', size=24)
        ax.text(160, 300, f'Spilled: {int(self.merged["cum_at"].max() - self.merged["cum_dt"].max())}', size=24)

        ax.legend()
        ax.grid()
            
    



        
class aircraft:
    def __init__(self, aircraft_id, intial_soc=80, flight_time=2, charge_time=1, soc_change=10, 
                soc_transition_time=np.array([0.0129,0.0133,0.0137,0.0142,0.0147,
                0.0153,0.0158,0.0166,0.0172,0.018,
                0.0188,0.0197,0.0207,0.0219,0.0231,
                0.0245,0.026,0.0278,0.03,0.0323,
                0.0351,0.0384,0.0423,0.0472,0.0536,
                0.0617,0.0726,0.0887,0.1136,0.1582,
                0.2622,0.9278,])*60):

        self.aircraft_id = aircraft_id
        self.soc = intial_soc
        self.delay_timestep = 0
        self.flight_time = flight_time
        self.charge_time = charge_time
        self.soc_change = soc_change
        self.soc_transition_allowed = []
        for i in range(len(soc_transition_time)):
            idx = i
            cnt = 0
            while soc_transition_time[i:idx+1].sum() <= 5:
                cnt += 1
                idx += 1
            self.soc_transition_allowed.append(cnt)
        self.soc_transition_allowed = np.array(self.soc_transition_allowed)
    
    def fly(self):
        self.delay_timestep += self.flight_time
        self.soc -= self.soc_change
    
    def charge(self):
        self.delay_timestep += self.charge_time
        # self.soc += self.soc_change
        self.soc += self.soc_transition_allowed[int(self.soc / 2.5)] * 2.5
    
    def update(self):
        if self.delay_timestep > 0:
            self.delay_timestep -= 1


class Vertiport:
    def __init__(self, aircrafts, print_log):
        self.aircrafts = aircrafts
        self.print_log = print_log

    def __get_idle_aircrafts__(self):
        idle_aircrafts = []
        for i in self.aircrafts:
            if (i.soc >= 10) & (i.delay_timestep == 0):
                idle_aircrafts.append(i)
            if self.print_log:
                print(f'Aircraft {i.aircraft_id} soc {i.soc}, delay {i.delay_timestep}')
        return len(idle_aircrafts)

    def update(self, num_flight, one_vertiport_system=False):
        for i in self.aircrafts:
            i.update()
        if self.print_log:
            print(f"Number of aircraft avalibale: {self.__get_idle_aircrafts__()}")
            print(f"Flight demand: {num_flight}")

        departed_aircrafts = []
        threshold = 5
        for i in self.aircrafts:
            if num_flight > 0:
                if (i.soc >= 10) & (i.delay_timestep == 0):
                    i.fly()
                    num_flight -= 1
                    departed_aircrafts.append(i)
                    if self.print_log:
                        print(i.aircraft_id, 'dispatched for flight', i.soc)
                elif (i.soc < 10) & (i.delay_timestep == 0):
                    i.charge()
                    if self.print_log:
                        print(i.aircraft_id, 'charging becaseu of low soc', i.soc)
        
            elif (len(self.aircrafts) > threshold) and (one_vertiport_system == False):
                if (i.soc >= 50) & (i.delay_timestep == 0):
                    i.fly()
                    departed_aircrafts.append(i)
                    threshold += 1
                elif i.delay_timestep == 0:
                    i.charge()

            elif (i.soc <= 70) & (i.delay_timestep == 0):
                i.charge()
                if self.print_log:
                    print(i.aircraft_id, "charging because of idling")
            


        if one_vertiport_system == False:
            for i in departed_aircrafts:
                self.aircrafts.remove(i)
        if self.print_log:
            print(f'NUmber of aircraft departed: {len(departed_aircrafts)}')

        return num_flight, departed_aircrafts

        
                
