from scipy.stats import skewnorm
import pandas as pd
import numpy as np
import warnings
import random
warnings.filterwarnings('ignore')

def generate_uam_schedule(lax_flight_arr, lax_flight_dep, total_lax_seats, auto_regressive_alpha, directional_demand, max_waiting_time=5):
    
    yearly_arr_capacity, yearly_dep_capacity = total_lax_seats
    # Calculate pax demand per scheduled flight
    lax_flight_arr['capacity'] = lax_flight_arr['capacity']/yearly_arr_capacity * directional_demand * 365
    lax_flight_dep['capacity'] = lax_flight_dep['capacity']/yearly_dep_capacity * directional_demand * 365

    lax_flight_arr = get_autoregressive_pax_count(lax_flight_arr, auto_regressive_alpha)
    lax_flight_dep = get_autoregressive_pax_count(lax_flight_dep, auto_regressive_alpha)


    lax_dtla = np.empty((1,))
    for i in range(lax_flight_arr.shape[0]):
        num_pax = lax_flight_arr.iloc[i,1]
        delta_t = skewnorm.rvs(3, loc=31, scale=np.sqrt(2*1.5**2), size=num_pax)
        delta_t += lax_flight_arr.iloc[i,0]
        lax_dtla = np.concatenate([lax_dtla, delta_t])
    lax_dtla[lax_dtla >= 1440] -= 1440
    lax_dtla = np.sort(lax_dtla)

    dtla_lax = np.empty((1,))
    for i in range(lax_flight_dep.shape[0]):
        num_pax = lax_flight_dep.iloc[i,1]   
        delta_t = skewnorm.rvs(3, loc=93, scale=40, size=num_pax)
        delta_t += np.random.normal(loc=10, scale=5/3, size=num_pax)
        delta_t = lax_flight_dep.iloc[i,0] - delta_t
        dtla_lax = np.concatenate([dtla_lax, delta_t])
    dtla_lax[dtla_lax < 0] += 1440
    dtla_lax = np.sort(dtla_lax)

    dtla_lax_sche, dtla_lax_num_pax_flight = build_schedules(dtla_lax, max_waiting_time)
    lax_dtla_sche, lax_dtla_num_pax_flight = build_schedules(lax_dtla, max_waiting_time)


    schedule = pd.DataFrame({'schedule': np.concatenate([lax_dtla_sche, dtla_lax_sche], axis=0),
            'od': np.concatenate([np.repeat('LAX_DTLA', len(lax_dtla_sche)), np.repeat('DTLA_LAX', len(dtla_lax_sche))], axis=0)})
    num_pax_per_flight = pd.DataFrame({'num_pax': np.concatenate([lax_dtla_num_pax_flight, dtla_lax_num_pax_flight], axis=0),
            'od': np.concatenate([np.repeat('LAX_DTLA', len(lax_dtla_sche)), np.repeat('DTLA_LAX', len(dtla_lax_sche))], axis=0)})

    pax_arrival_times = np.concatenate([np.round(np.sort(dtla_lax)*60), np.round(np.sort(lax_dtla)*60)], axis=0)
    passenger_id = np.arange(0, len(pax_arrival_times))
    origin = np.concatenate([np.repeat('DTLA', len(dtla_lax)), np.repeat('LAX', len(lax_dtla))], axis=0)
    destination = np.concatenate([np.repeat('LAX', len(dtla_lax)), np.repeat('DTLA', len(lax_dtla))], axis=0)
    paxArrivalDf = pd.DataFrame({'passenger_id':passenger_id, 'passenger_arrival_time':pax_arrival_times, 'origin_vertiport_id':origin, 'destination_vertiport_id':destination})

    return schedule, paxArrivalDf, num_pax_per_flight


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

def get_autoregressive_pax_count(arr_merged, auto_regressive_alpha):
    # Calculate hourly rate
    arr_merged_v2 = arr_merged.copy()
    arr_merged_v2['time_interval'] = arr_merged_v2['time'] // 60
    arr_merged_v2_grouped = arr_merged_v2.groupby('time_interval').sum('capacity')
    arr_merged_v2_grouped = arr_merged_v2_grouped.merge(pd.DataFrame({'time_interval': np.arange(0, 24, 1), 'demand':0}), on='time_interval', how='outer').sort_values('time_interval').reset_index(drop=True).fillna(0)
    
    # Calculate flight pax density over hourly rate
    arr_merged_v2_merged = arr_merged_v2.merge(arr_merged_v2_grouped, on='time_interval')
    arr_merged_v2_merged['density'] = arr_merged_v2_merged['capacity_x'] / arr_merged_v2_merged['capacity_y']
    arr_merged_v2_merged['cum_density'] = arr_merged_v2_merged.groupby('time_interval')['density'].cumsum()

    # Generate one realization of hourly rate
    realized_rate = auto_regressive_poisson(arr_merged_v2_grouped['capacity'].values, alpha=auto_regressive_alpha)

    # Assign hourly rate to each flight by inverse cdf
    assigned_pax = np.empty(shape=(1,), dtype=int)
    for i in range(24):
        check = arr_merged_v2_merged[arr_merged_v2_merged['time_interval'] == i]
        if check.shape[0] == 0:
            continue
        x = inverse_cdf(check['cum_density'].values, num_samples=realized_rate[i])
        assigned_pax_i = np.histogram(x, np.arange(0, check.shape[0]+1))[0]
        assigned_pax = np.concatenate([assigned_pax, assigned_pax_i])

    assigned_pax = assigned_pax[1:]
    output = pd.DataFrame({'time':arr_merged['time'], 'capacity':assigned_pax})

    return output


def auto_regressive_poisson(rate, alpha):
    lambda_0_0 = rate[0]
    x_0 = np.random.poisson(lambda_0_0)
    output = [x_0]
    for i in range(1,len(rate)):
        lambda_1_0 = rate[i]

        if lambda_1_0 == 0:
            x_0 = 0
        else:
            if lambda_0_0 == 0:
                lambda_1 = lambda_1_0
            else:
                lambda_1 = lambda_1_0 + (x_0 - lambda_0_0)*lambda_1_0/lambda_0_0 * alpha
            x_0 = np.random.poisson(lambda_1)
            
        lambda_0_0 = lambda_1_0
        output.append(x_0)

    return np.array(output)


def inverse_cdf(cumulative_pmf, num_samples):
    samples = []
    for _ in range(num_samples):
        u = np.random.uniform(0, 1)
        sample_index = (u <= cumulative_pmf)
        samples.append(np.argmax(sample_index))

    samples = np.array(samples)
    return samples