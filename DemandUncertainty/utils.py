from scipy.stats import skewnorm
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def build_schedules(queue, max_waiting_time, occupancy=4):
    queue = np.sort(queue)
    departure_time_aircraft = []
    num_pax_per_flight = []
    cnt = 0
    first_pax = queue[0]
    for i in range(queue.shape[0]):
        cnt += 1
        if (queue[i]-first_pax) > max_waiting_time:
            departure_time_aircraft.append(first_pax+max_waiting_time)
            num_pax_per_flight.append(cnt-1)
            cnt = 1
            first_pax = queue[i]

        elif cnt == occupancy:
            departure_time_aircraft.append(queue[i])
            num_pax_per_flight.append(occupancy)
            cnt = 0
            try:
                first_pax = queue[i+1]
            except IndexError:
                pass

    if queue[i] > departure_time_aircraft[-1]:
        departure_time_aircraft.append(np.max(queue))
        num_pax_per_flight.append(cnt)

    departure_time_aircraft = np.array(departure_time_aircraft)
    num_pax_per_flight = np.array(num_pax_per_flight)

    return departure_time_aircraft, num_pax_per_flight


def generate_uam_schedule(lax_flight_arr, lax_flight_dep, seatcap, directional_demand=1500, max_waiting_time=5):
    # Set seat capacity mapping
    seat_capacity = {'B752':170, 'B748':300, 'E75L':60, 'B77W':300, 'B739':170, 'A321':170, 'B737':170, 'B763':300,
       'B744':300, 'B772':300, 'B738':170, 'B753':170, 'B789':300, 'B77L':300, 'A388':300, 'A332':300,
       'CRJ2':60, 'B712':60, 'B788':300, 'CRJ7':60, 'E75S':60, 'A320':170, 'A319':170, 'A359':300,
       'E55P':60, 'B764':300, 'E190':60, 'A333':300, 'C680':0, 'A21N':170, 'F2TH':0,
       'A343':300, 'C208':0, 'B350':300, 'E50P':0, 'PC12':0, 'B78X':300, 'C56X':0, 'LJ60':0,
       'CL35':0, 'A20N':170, 'B732':170}
    seat_capacity = pd.DataFrame.from_dict(seat_capacity, orient='index').reset_index()
    seat_capacity.columns=['ETMS_EQPT', 'capacity']

    lax_flight_arr['TAILNO'] = lax_flight_arr['TAILNO'].str.strip()
    lax_flight_dep['TAILNO'] = lax_flight_dep['TAILNO'].str.strip()

    lax_flight_arr = lax_flight_arr.merge(seatcap[['TAIL_NUMBER', 'NUMBER_OF_SEATS']], 
                                        left_on='TAILNO', right_on='TAIL_NUMBER',
                                        how='left')
    lax_flight_dep = lax_flight_dep.merge(seatcap[['TAIL_NUMBER', 'NUMBER_OF_SEATS']], 
                                    left_on='TAILNO', right_on='TAIL_NUMBER',
                                    how='left')
    
    arr_merged = lax_flight_arr.merge(seat_capacity, on='ETMS_EQPT', how='left')
    arr_merged['capacity'] = arr_merged['NUMBER_OF_SEATS'].fillna(arr_merged['capacity'])
    arr_merged = arr_merged[['time', 'capacity']].dropna()

    dep_merged = lax_flight_dep.merge(seat_capacity, on='ETMS_EQPT', how='left')
    dep_merged['capacity'] = dep_merged['NUMBER_OF_SEATS'].fillna(dep_merged['capacity'])
    dep_merged = dep_merged[['time', 'capacity']].dropna()

    # Calculate pax demand per scheduled flight
    arr_factor = directional_demand/arr_merged['capacity'].sum()
    arr_merged['capacity'] = arr_merged['capacity']*arr_factor
    dep_factor = directional_demand/dep_merged['capacity'].sum()
    dep_merged['capacity'] = dep_merged['capacity']*dep_factor

    # Generate pax arrival times at vertiport
    # np.random.seed(42)
    lax_dtla = np.empty((1,))
    for i in range(arr_merged.shape[0]):
        lambda_i = arr_merged.iloc[i,1]
        num_pax = np.random.poisson(lambda_i)
        delta_t = skewnorm.rvs(3, loc=31, scale=np.sqrt(2*1.5**2), size=num_pax)
        delta_t += arr_merged.iloc[i,0]
        lax_dtla = np.concatenate([lax_dtla, delta_t])
    lax_dtla[lax_dtla >= 1440] -= 1440
    lax_dtla = np.sort(lax_dtla)

    dtla_lax = np.empty((1,))
    for i in range(dep_merged.shape[0]):
        lambda_i = dep_merged.iloc[i,1]
        num_pax = np.random.poisson(lambda_i)
        delta_t = skewnorm.rvs(3, loc=93, scale=40, size=num_pax)
        delta_t += np.random.normal(loc=10, scale=5/3, size=num_pax)
        delta_t = dep_merged.iloc[i,0] - delta_t
        dtla_lax = np.concatenate([dtla_lax, delta_t])
    dtla_lax[dtla_lax < 0] += 1440
    dtla_lax = np.sort(dtla_lax)

    dtla_lax_sche, dtla_lax_num_pax_flight = build_schedules(dtla_lax, max_waiting_time)
    lax_dtla_sche, lax_dtla_num_pax_flight = build_schedules(lax_dtla, max_waiting_time)


    schedule = pd.DataFrame({'schedule': np.concatenate([lax_dtla_sche, dtla_lax_sche], axis=0),
            'od': np.concatenate([np.repeat('LAX_DTLA', len(lax_dtla_sche)), np.repeat('DTLA_LAX', len(dtla_lax_sche))], axis=0)})

    return schedule

def pois_generate(rate, alpha):
    lambda_0_0 = rate[0]
    x_0 = np.random.poisson(lambda_0_0)
    output = [x_0]
    for i in range(1,len(rate)):
        lambda_1_0 = rate[i]
        lambda_1 = lambda_1_0 + (x_0 - lambda_0_0)*lambda_1_0/lambda_0_0 * alpha

        x_0 = np.random.poisson(lambda_1)
        lambda_0_0 = lambda_1_0
        output.append(x_0)

    return output