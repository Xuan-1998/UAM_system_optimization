import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class system:
    def __init__(self, fleet_size, num_pax_flight_path, flight_schedule_path):
        self.fleet_size = fleet_size
        num_pax_flight = pd.read_csv(num_pax_flight_path)
        flight_sche = pd.read_csv(flight_schedule_path)
        schedule = pd.concat([flight_sche, num_pax_flight['num_pax']], axis=1)
        schedule['schedule'] = np.ceil(schedule['schedule'] / 5)
        self.schedule = schedule
        self.vertiports = [Vertiport([aircraft() for i in range(fleet_size//2)]), Vertiport([aircraft() for i in range(fleet_size-fleet_size//2)])]

        self.arrival_curve = schedule.groupby('schedule').sum()['num_pax'].reset_index()

    def __update__(self, num_flight, num_pax):
        print(f'------------------------------------------timestep------------------------------------------')

        num_flight_v1, departed_aircraft_v1v2 = self.vertiports[0].update(num_flight[0], num_flight[1], self.vertiports[1])
        num_flight_v2, departed_aircraft_v2v1 = self.vertiports[1].update(num_flight[1], num_flight[0], self.vertiports[0])
        
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
    def __init__(self, intial_soc=80):
        self.soc = intial_soc
        self.delay_timestep = 0
    
    def fly(self):
        self.delay_timestep += 2
        self.soc -= 10
    
    def charge(self):
        self.delay_timestep += 1
        self.soc += 10
    
    def update(self):
        if self.delay_timestep > 0:
            self.delay_timestep -= 1


class Vertiport:
    def __init__(self, aircrafts):
        self.aircrafts = aircrafts

    def __get_idle_aircrafts__(self):
        idle_aircrafts = []
        for i in self.aircrafts:
            if (i.soc >= 10) & (i.delay_timestep == 0):
                idle_aircrafts.append(i)
            print(f'Idle aircrafts soc {i.soc}, delay {i.delay_timestep}')
        return len(idle_aircrafts)

    def update(self, num_flight, num_flight_at_other_vertiport, other_vertiports):
        # free_aircraft_other_vertiport = other_vertiports.__get_idle_aircrafts__()

        for i in self.aircrafts:
            i.update()

        print(f"Number of aircraft avalibale: {self.__get_idle_aircrafts__()}")
        print(f"Flight demand: {num_flight}")

        departed_aircrafts = []
        for i in self.aircrafts:
            if (i.soc >= 10) & (i.delay_timestep == 0):

                if num_flight > 0:
                    i.fly()
                    num_flight -= 1
                    departed_aircrafts.append(i)

                # elif num_flight_at_other_vertiport > 0:
                #     num_flight = num_flight_at_other_vertiport
                #     i.fly()
                #     num_flight -= 1
                #     departed_aircrafts.append(i)
            
            elif (i.soc == 0) & (i.delay_timestep == 0):
                i.charge()

        for i in departed_aircrafts:
            self.aircrafts.remove(i)

        print(f'NUmber of aircraft departed: {len(departed_aircrafts)}')

        return num_flight, departed_aircrafts

        
                
