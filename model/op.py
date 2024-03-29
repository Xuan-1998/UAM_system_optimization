from gurobipy import *
import pandas as pd
import numpy as np
import ast
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
font = {'size'   : 24}
matplotlib.rc('font', **font)


class FleetSizeOptimizer:
    def __init__(self, flight_time, energy_consumption, schedule, fixed_cost=1, variable_cost=0.0000001, time_step=5):
        # Set up time-related parameters
        self.time_step = time_step 
        self.schedule_time_step = int(1440/time_step)

        # Flight time and energy consumption
        self.flight_time = np.ceil(flight_time/time_step).astype(int)
        self.soc_transition_time = np.array([0.0129,0.0133,0.0137,0.0142,0.0147,
                                0.0153,0.0158,0.0166,0.0172,0.018,
                                0.0188,0.0197,0.0207,0.0219,0.0231,
                                0.0245,0.026,0.0278,0.03,0.0323,
                                0.0351,0.0384,0.0423,0.0472,0.0536,
                                0.0617,0.0726,0.0887,0.1136,0.1582,
                                0.2622,0.9278,])*60
        self.energy_consumption = np.ceil(energy_consumption/(80/len(self.soc_transition_time))).astype(int)
        # Check the dimension of flight time and energy consumption. Make sure they are time-varying.
        if len(self.flight_time.shape) != len(self.energy_consumption.shape):
            raise DimensionError('The dimension of the flight time array must match the dimension of the energy consumption array')
        if len(self.flight_time.shape) == 2:
            self.flight_time = np.repeat(self.flight_time[np.newaxis,:,:], self.schedule_time_step+1, axis=0)
        if len(self.energy_consumption.shape) == 2:
            self.energy_consumption = np.repeat(self.energy_consumption[np.newaxis,:,:], self.schedule_time_step+1, axis=0)  

        # Set up fixed and variable cost
        self.fixed_cost = fixed_cost
        self.variable_cost = variable_cost

        # Get constants
        self.K = len(self.soc_transition_time)
        self.T = int(np.max(self.flight_time)) + self.schedule_time_step + 1
        self.V = [i for i in range(self.energy_consumption.shape[1])]

        # Get flight values
        self.f_values = np.zeros((self.T, len(self.V), len(self.V)))
        data = pd.read_csv(f'input/{schedule}')
        self.flight_schedule = data
        bins = np.arange(0, 24*60+1, time_step)
        data['time_bins'] = pd.cut(data['schedule'], bins, right=False)
        counts = data.groupby(['od', 'time_bins']).size().unstack('od', fill_value=0)
        LAX_DTLA = np.array(counts['LAX_DTLA'].tolist())
        DTLA_LAX = np.array(counts['DTLA_LAX'].tolist())
        for t in range(self.T-self.flight_time.max()-1):
            self.f_values[t+1][0][1] = LAX_DTLA[t]
            self.f_values[t+1][1][0] = DTLA_LAX[t]


    def optimize(self, output_path, 
                 spill_optimization=False, occupancy=None, seat_capacity=None, fleet_size=None,
                 charging_station=[True, True], number_of_pads=None, 
                 verbose=True, optimality_gap=0.05):
        # Load the parameters
        T = self.T
        K = self.K
        V = self.V
        gamma = self.soc_transition_time
        schedule_time_step = self.schedule_time_step
        fixed_cost = self.fixed_cost
        variable_cost = self.variable_cost
        f_values = self.f_values
        max_flight_time = int(np.max(self.flight_time))

        tau = []
        kappa = []
        for _ in range(T):
            tau_second_layer = []
            kappa_second_layer = []
            for _ in range(len(V)):
                tau_third_layer = []
                kappa_third_layer = []
                for _ in range(len(V)):
                    tau_innermost_layer = [0]
                    kappa_innermost_layer = [0]
                    tau_third_layer.append(tau_innermost_layer)
                    kappa_third_layer.append(kappa_innermost_layer)

                tau_second_layer.append(tau_third_layer)
                kappa_second_layer.append(kappa_third_layer)
            tau.append(tau_second_layer)
            kappa.append(kappa_second_layer)

        # tau here is a mapping from t to the time an aircraft leaves from the origin
        for t in range(1, self.schedule_time_step+1):
            for i in V:
                for j in V:
                    if i != j:
                        tau[t+self.flight_time[t, i, j]][i][j].append(t)
                        kappa[t+self.flight_time[t, i, j]][i][j].append(self.energy_consumption[t, i, j])

        # Create the spill parameter if running spill optimization
        if spill_optimization:
            pij = np.zeros((T, 2, 2))
            bins = np.arange(0, 24*60+1, 5)
            merged = pd.concat([self.flight_schedule, occupancy['num_pax']], axis=1)
            merged['time_bins'] = pd.cut(merged['schedule'], bins, right=False)
            pax_count = merged.groupby(['od', 'time_bins'])['num_pax'].sum().unstack('od', fill_value=0)
            LAX_DTLA = np.array(pax_count['LAX_DTLA'].tolist())
            DTLA_LAX = np.array(pax_count['DTLA_LAX'].tolist())

            for t in range(T-max_flight_time-1):
                pij[t+1][0][1] = LAX_DTLA[t] # get the first (and only) item of the inner list
                pij[t+1][1][0] = DTLA_LAX[t] # get the first (and only) item of the inner list

        ########################################Optimization Model##############################################
        m = Model("FleetSizeOptimizer")
        if verbose == False:
            m.setParam('OutputFlag', 0)
        m.setParam('threads', 2)
   
        # Create variables
        ni = m.addVars(((t, i, k) for t in range(T) for i in V for k in range(K+1)), vtype=GRB.INTEGER, name="n")
        uij = m.addVars(((t, i, j, k) for t in range(T) for i in V for j in V for k in range(K+1) if i != j), vtype=GRB.INTEGER, name="u")
        cijk = m.addVars(((t, i, x, y) for t in range(T) for i in V for x in range(K+1) for y in range(K+1) if x < y), vtype=GRB.INTEGER, name="c")
        if spill_optimization:
            sij = m.addVars(((t, i, j) for t in range(T) for i in V for j in V  if i!=j), vtype=GRB.INTEGER, name="s")

        # Objective function
        if spill_optimization:
            m.setObjective(fixed_cost*sij.sum('*', '*', '*'), GRB.MINIMIZE)
        else:
            m.setObjective(fixed_cost*(ni.sum(0, '*', '*') + 
                        uij.sum(0, '*', '*', '*') + 
                        cijk.sum(0, '*', '*', '*')) + 
                        variable_cost*(uij.sum('*', '*', '*', '*')), GRB.MINIMIZE)
                        
        # Constraint 1: Dynamic equation
        for i in V:
            for k in range(K+1):
                for t in range(1, T):
                    m.addConstr(
                        ni[t, i, k] == ni[t-1, i, k] + 
                        quicksum(uij[tau[t][j][i][index], j, i, k + kappa[t][j][i][index]] for j in V for index in range(len(tau[t][j][i])) if j != i and tau[t][j][i][index] != 0 and k + kappa[t][j][i][index] <= K) -
                        quicksum(uij[t, i, j, k] for j in V if j != i) +
                        quicksum(cijk[t-np.ceil(sum(gamma[x:k])/self.time_step), i, x, k] for x in range(k) if t-np.ceil(sum(gamma[x:k])/1) >= 0) -
                        quicksum(cijk[t, i, k, y] for y in range(k+1, K+1))
                    )

        # Constraint 2: Cyclo-stationarity Constraint
        for k in range(K+1):
            for i in V:
                for j in V:
                    if i != j:
                        m.addConstr(uij[0, i, j, k] == uij[T-1, i, j, k])
                m.addConstr(ni[0, i, k] == ni[T-1, i, k])

        for x in range(K+1):
            for y in range(K+1):
                for i in V:
                    if (x < y):
                        m.addConstr(cijk[0, i, x, y] == cijk[T-1, i, x, y])

        # Constraint 3: Demand Constraint
        if spill_optimization:
            # With spill optimizaiton, demand is not guaranteed to be met
            # Constraint 3.1: Spill constaint for max function
            for i in V:
                for j in V:
                    for t in range(T):
                        if i != j:
                            m.addConstr(sij[t, i, j] >= 0)
                            m.addConstr(sij[t, i, j] >= pij[t, i, j] - seat_capacity*uij.sum(t, i, j, '*'))
            # Constaint 3.2 Fleet size constraint
            m.addConstr(ni.sum(0, '*', '*') + uij.sum(0, '*', '*', '*') + cijk.sum(0, '*', '*', '*') == fleet_size)
        else:
            # Constraint 3: Demand Constraint
            for j in V:
                for i in V:
                    if i != j:
                        for t in range(T-1):
                            m.addConstr(uij.sum(t, i, j, '*') >= f_values[t][i][j])

        # Constraint 4: Energy Constraint
        for i in V:
            for j in V:
                for t in range(T):
                    if i != j:
                        m.addConstr(uij[t, i, j, 0] == 0)
        
        # Constraint 5: Charging Station Constraint
        if len(charging_station) != self.f_values.shape[1]:
            raise ValueError("The binary list indicating whether or not charging infrastrcture exist must be the same length as the number of vertiports")

        for idx, i in enumerate(charging_station):
            if i == False:
                m.addConstr(cijk.sum('*', idx, '*', '*') == 0)

        row, col, TPRIME = self.__getVars__()

        # Constraint 6: Parking Facility Constraint
        if number_of_pads is not None:
            for i in V:
                for t in range(T):
                    m.addConstr((ni.sum(t, i, '*') + 
                    quicksum(cijk[time_adjusted, 
                                i, 
                                row[index], 
                                col[index]] 
                                for index in range(len(col))
                                for time_adjusted in range(t-int(TPRIME[row[index], col[index]]), t)
                                if t >= TPRIME[row[index], col[index]])+
                    cijk.sum(t, i, '*', '*'))  <= number_of_pads[i])

        m.update()
        m.Params.MIPGap = optimality_gap
        m.Params.FeasibilityTol = 1e-7
        m.optimize()

        # Collect results
        if spill_optimization:
            total_spill = int(sij.sum('*', '*', '*').getValue())
            print('The total spill is:', total_spill, 'Fleet Size:', fleet_size)
            with open('output/'+output_path+'_total_spill.txt', 'w') as file:
                file.write('Total Spill: ' + str(total_spill))
            with open('output/'+output_path+'_spill_op_result.txt', 'w') as file:
                old_stdout = sys.stdout
                sys.stdout = file
                print("results")
                for v in m.getVars():
                    if v.x > 0: 
                        print('{} = {}'.format(v.varName, v.x))
                sys.stdout = old_stdout
            print('The total spill is:', total_spill)
            return total_spill
        else:
            total_fleet_size = int(ni.sum(0, '*', '*').getValue() + uij.sum(0, '*', '*', '*').getValue() + cijk.sum(0, '*', '*', '*').getValue())
            with open('output/'+output_path+'_fleetsize.txt', 'w') as file:
                file.write('Total Fleet Size: ' + str(total_fleet_size))

            with open('output/'+output_path+'_op_result.txt', 'w') as file:
                old_stdout = sys.stdout
                sys.stdout = file
                print("results")
                for v in m.getVars():
                    if v.x > 0: 
                        print('{} = {}'.format(v.varName, v.x))
                sys.stdout = old_stdout
            return total_fleet_size

    def __getVars__(self):
        w = np.zeros(shape=(self.K+1, self.K+1))
        for i in range(self.K+1):
            for j in range(self.K+1):
                if j > i:
                    w[i,j] = self.soc_transition_time[i:j].sum()
        w = w // 5 + 1
        row, col = np.where(w > 1)

        return row, col, w-1
        

    def parse_result(self, output_file):
        results = pd.read_table(f'output/{output_file}')

        n = results[results["results"].str.contains("n")]
        u = results[results["results"].str.contains("u")]
        c = results[results["results"].str.contains("c")]
        varn = n['results'].str.split('=').apply(lambda x: x[0])
        valn = n['results'].str.split('=').apply(lambda x: x[1])
        t = varn.str.split('n').apply(lambda x: ast.literal_eval(x[1])).apply(lambda x: x[0])
        i = varn.str.split('n').apply(lambda x: ast.literal_eval(x[1])).apply(lambda x: x[1])
        k = varn.str.split('n').apply(lambda x: ast.literal_eval(x[1])).apply(lambda x: x[2])
        amount = valn
        specificn = pd.DataFrame(np.array([t,i,k,amount]).T).reset_index(drop=True)
        specificn.columns = ['t','i', 'k', 'amount']
        specificn['name'] = 'n'


        varu = u['results'].str.split('=').apply(lambda x: x[0])
        valu = u['results'].str.split('=').apply(lambda x: x[1])
        t = varu.str.split('u').apply(lambda x: ast.literal_eval(x[1])).apply(lambda x: x[0])
        i = varu.str.split('u').apply(lambda x: ast.literal_eval(x[1])).apply(lambda x: x[1])
        j = varu.str.split('u').apply(lambda x: ast.literal_eval(x[1])).apply(lambda x: x[2])
        k = varu.str.split('u').apply(lambda x: ast.literal_eval(x[1])).apply(lambda x: x[3])
        amount = valu
        specificu = pd.DataFrame(np.array([t,i,j,k,amount]).T).reset_index(drop=True)
        specificu.columns = ['t','i', 'j', 'k', 'amount']
        specificu['name'] = 'u'

        varc = c['results'].str.split('=').apply(lambda x: x[0])
        valc = c['results'].str.split('=').apply(lambda x: x[1])
        t = varc.str.split('c').apply(lambda x: ast.literal_eval(x[1])).apply(lambda x: x[0])
        i = varc.str.split('c').apply(lambda x: ast.literal_eval(x[1])).apply(lambda x: x[1])
        x = varc.str.split('c').apply(lambda x: ast.literal_eval(x[1])).apply(lambda x: x[2])
        y = varc.str.split('c').apply(lambda x: ast.literal_eval(x[1])).apply(lambda x: x[3])
        amount = valc
        specificc = pd.DataFrame(np.array([t,i,x,y,amount]).T).reset_index(drop=True)
        specificc.columns = ['t','i', 'x', 'y', 'amount']
        specificc['name'] = 'c'
        specificn['amount'] = specificn['amount'].astype(float)
        specificu['amount'] = specificu['amount'].astype(float)
        specificc['amount'] = specificc['amount'].astype(float)

        self.specificn = specificn
        self.specificu = specificu
        self.specificc = specificc


    def __compute_states__(self, specificc, specificu, specificn):
        end = int(1440/self.time_step) + 1 + int(np.max(self.flight_time))
        all_c = np.zeros(shape=(1,end), dtype=int)
        for i in range(specificc.shape[0]):
            val = int(specificc['amount'][i])
            soc0 = int(specificc['x'][i])
            soc1 = int(specificc['y'][i])
            t = int(specificc['t'][i])
            time_charge = int(np.ceil(self.soc_transition_time[soc0: soc1].sum()/self.time_step))
            occupied = np.zeros(shape=(val,end))
            for j in range(val):
                occupied[j][t:t+time_charge] = 1

            all_c = np.concatenate([all_c, occupied], axis=0)
        all_c = all_c[1:,:]

        all_u = np.zeros(shape=(1,end), dtype=int)
        for i in range(specificu.shape[0]):
            val = int(specificu['amount'][i])
            t = int(specificu['t'][i])
            origin = int(specificu['i'][i])
            dest = int(specificu['j'][i])
            
            flight = np.zeros(shape=(val,end))
            for j in range(val):
                flight[j][t:t+self.flight_time[t, origin, dest]] = 1

            all_u = np.concatenate([all_u, flight], axis=0)
        all_u = all_u[1:,:]

        all_n = np.zeros(shape=(1,end), dtype=int)
        for i in range(specificn.shape[0]):
            val = int(specificn['amount'][i])
            t = int(specificn['t'][i])
            idle = np.zeros(shape=(val,end))
            for j in range(val):
                idle[j][t] = 1
            all_n = np.concatenate([all_n, idle], axis=0)
        all_n = all_n[1:,:]
    
        return all_c, all_u, all_n


    def calculate_aircraft_states(self, idx='all'):
        self.total_fleet_size = self.specificc[self.specificc['t'] == 0]['amount'].sum() + self.specificu[self.specificu['t'] == 0]['amount'].sum() + self.specificn[self.specificn['t'] == 0]['amount'].sum()

        if idx == 'all':
            self.all_c, self.all_u, self.all_n = self.__compute_states__(self.specificc, self.specificu, self.specificn)
        else:
            filtered_specificc = self.specificc[self.specificc['i'] == idx].reset_index(drop=True)
            filtered_specificu = self.specificu[self.specificu['i'] == idx].reset_index(drop=True)
            filtered_specificn = self.specificn[self.specificn['i'] == idx].reset_index(drop=True)
            filtered_all_c, filtered_all_u, filtered_all_n = self.__compute_states__(filtered_specificc, filtered_specificu, filtered_specificn)

            return filtered_all_c, filtered_all_u, filtered_all_n

    def plot_aircraft_state(self):
        x0 = 0
        x1 = self.T
        fig, ax = plt.subplots(figsize=(20,8), dpi=200)
        sns.lineplot(self.all_c.sum(axis=0)[x0:x1], label='Charging', ax=ax)
        sns.lineplot(self.all_u.sum(axis=0)[x0:x1], label='In Flight', ax=ax)
        sns.lineplot(self.all_n.sum(axis=0)[x0:x1], label='Idling', ax=ax)
        sns.lineplot((self.all_c.sum(axis=0)[x0:x1] + self.all_n.sum(axis=0)[x0:x1] + self.all_u.sum(axis=0)[x0:x1]), label='All Aircraft', ax=ax)
        ax.set(title='State of Aircraft',
            ylabel='Number of Aircrafts',
            xticks=np.concatenate([np.array([0,1]), np.arange(24,300, 12*2)]),
            xticklabels=['']+[str(i)+':00' for i in range(0,25,2)])
        plt.legend(loc='upper left', prop={'size': 16})
        plt.grid();
    
    def get_summary_statistics(self, flight_miles, return_summary=False):
        print(f"Fleet size: {self.total_fleet_size}")

        vertiport_specific_pads = np.zeros(shape=(len(self.V), ), dtype=int)
        for i in self.V:
            filtered_all_c, filtered_all_u, filtered_all_n = self.calculate_aircraft_states(idx=i)
            vertiport_specific_pads[i] = np.max(filtered_all_c.sum(axis=0) + filtered_all_n.sum(axis=0)).astype(int)


        print(f"Total number of pads: {np.sum(vertiport_specific_pads)}; {vertiport_specific_pads} ")
        print(f"Total number of flights: {self.specificu['amount'].sum()}; demand: {self.f_values.sum()}; repositioning: {self.specificu['amount'].sum()-self.f_values.sum()}")


        energy_consumption = 0
        for t in range(1, self.schedule_time_step+1):
            actual_flight_number_at_t = np.zeros(shape=(len(self.V), len(self.V)))
            for i in self.V:
                for j in self.V:
                    if i != j:
                        actual_flight_number_at_t[i][j] = self.specificu[(self.specificu['i'] == i) & (self.specificu['j'] == j) & (self.specificu['t'] == t)]['amount'].sum()

            energy_consumption_at_t = actual_flight_number_at_t @ self.energy_consumption[t].T
            energy_consumption += energy_consumption_at_t.diagonal().sum()

        print(f"Total energy consumption: {energy_consumption} kWh")

        actual_flight_number = np.zeros(shape=(len(self.V), len(self.V)))
        for i in self.V:
            for j in self.V:
                if i != j:
                    actual_flight_number[i][j] = self.specificu[(self.specificu['i'] == i) & (self.specificu['j'] == j)]['amount'].sum()

        total_aircraft_miles = actual_flight_number @ flight_miles.T
        print(f"Total aircraft miles: {total_aircraft_miles.diagonal().sum()} mi")

        revenue_aircraft_miles = self.f_values.sum(axis=0) @ flight_miles.T
        print(f"Total revenue aircraft miles: {revenue_aircraft_miles.diagonal().sum()}")
        print(f"Ratio of revenue aircraft miles to aircraft miles: {revenue_aircraft_miles.diagonal().sum()/total_aircraft_miles.diagonal().sum()}")

        if return_summary:
            summary_dict = {
                'fleet_size': self.total_fleet_size,
                'total_pads': np.sum(vertiport_specific_pads),
                'pads': vertiport_specific_pads,
                'number_of_flights': self.specificu['amount'].sum(),
                'number_of_repositioning_flights': self.specificu['amount'].sum()-self.f_values.sum(),
                'energy_consumption': energy_consumption.sum(),
                'TAM': total_aircraft_miles.diagonal().sum(),
                'RAM': revenue_aircraft_miles.diagonal().sum()
            }
            return summary_dict




    



