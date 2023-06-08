from gurobipy import *
import pandas as pd
import numpy as np



def number_aircrafts_lp(schedule, 
                        schedule_time_step,
                        output_path,
                        tau=[[0, 5.92], [5.85, 0]], 
                        kappa = [[0, 7.71875], [7.44375, 0]], 
                        gamma = [1.567183013,1.670689686,1.79349788,1.935972287,2.103057098,
                                 2.30172949,2.541890384,2.83806663,3.212473781,3.70088931,
                                 4.364896382,5.32037536,6.814736187,9.490547548,15.74119426,55.66984127], 
                        fixed_cost=1, 
                        variable_cost=0):
    # Flight time matrix
    tau = np.array(tau) / 5
    tau = np.ceil(tau)

    # SOC levels drop matrix
    kappa = np.array(kappa) / (80/len(gamma))
    kappa = np.ceil(kappa)

    # Charging Time matrix
    gamma = np.array(gamma) 

    max_flight_time = int(np.max(tau))

    T = schedule_time_step + 1 + max_flight_time

    # Constants
    K = len(gamma)
    V = [0, 1]

    f_values = np.zeros((T, 2, 2))
    data = pd.read_csv(f'../input/{schedule}.csv')
    # Create 5 minute bins (24 hours * 60 minutes / 5 minute intervals)
    bins = np.arange(0, 24*60+1, 5)

    # Bin the schedule data
    data['time_bins'] = pd.cut(data['schedule'], bins, right=False)

    # Group by od and time_bins, count the number of occurrences, unstack 'od' and fill NaNs with 0
    counts = data.groupby(['od', 'time_bins']).size().unstack('od', fill_value=0)

    # Get the two 288 element lists
    LAX_DTLA = np.array(counts['LAX_DTLA'].tolist())
    DTLA_LAX = np.array(counts['DTLA_LAX'].tolist())

    for t in range(T-max_flight_time-1):
        f_values[t+1][0][1] = LAX_DTLA[t] # get the first (and only) item of the inner list
        f_values[t+1][1][0] = DTLA_LAX[t] # get the first (and only) item of the inner list

    # Create a new model
    m = Model("Vertiport_Aircraft_Routing")

    # Create variables
    ni = m.addVars(((t, i, k) for t in range(T) for i in V for k in range(K+1)), vtype=GRB.INTEGER, name="n")
    uij = m.addVars(((t, i, j, k) for t in range(T) for i in V for j in V for k in range(K+1) if i != j), vtype=GRB.INTEGER, name="u")
    cijk = m.addVars(((t, i, x, y) for t in range(T) for i in V for x in range(K+1) for y in range(K+1) if x < y), vtype=GRB.INTEGER, name="c")

    m.setObjective(fixed_cost*(ni.sum(0, '*', '*') + 
                   uij.sum(0, '*', '*', '*') + 
                   cijk.sum(0, '*', '*', '*')) + 
                   variable_cost*(uij.sum('*', '*', '*', '*')-LAX_DTLA.sum()-DTLA_LAX.sum()), GRB.MINIMIZE)

    # Dynamic equation
    for i in V:
        for k in range(K+1):
            for t in range(1, T):
                m.addConstr(
                    ni[t, i, k] == ni[t-1, i, k] + 
                    quicksum(uij[t-tau[j][i], j, i, k+kappa[j][i]] for j in V if j != i and t-1-tau[j][i] >= 0 and k+kappa[j][i] <= K) -
                    quicksum(uij[t, i, j, k] for j in V if j != i) +
                    quicksum(cijk[t-np.ceil(sum(gamma[x:k])/5), i, x, k] for x in range(k) if t-np.ceil(sum(gamma[x:k])/5) >= 0) -
                    quicksum(cijk[t, i, k, y] for y in range(k+1, K+1))
                )

    # Stationary Constraint
    for k in range(K+1):
        for i in V:
            m.addConstr(ni[0, i, k] == ni[schedule_time_step+max_flight_time, i, k])
            m.addConstr(uij[0, i, 1-i, k] == uij[schedule_time_step+max_flight_time, i, 1-i, k])
            
    for x in range(K+1):
        for y in range(K+1):
            for i in V:
                if (x < y):
                    m.addConstr(cijk[0, i, x, y] == cijk[schedule_time_step+max_flight_time, i, x, y])
                    
    for j in V:
        for i in V:
            if i != j:
                for t in range(T-1):
                    m.addConstr(uij.sum(t, i, j, '*') >= f_values[t][i][j])

    # Can't fly when SOC = 0
    for i in V:
        for j in V:
            for t in range(T):
                if i != j:
                    m.addConstr(uij[t, i, j, 0] == 0)

    # Integrate new variables
    m.update()

    m.Params.MIPGap = 0.05  # Set the optimality tolerance to 5%
    m.Params.FeasibilityTol = 1e-7
    # Solve model
    m.optimize()

    # Print results
    for v in m.getVars():
        if v.x > 0:  # Print only non-zero variables for clarity
            print('{} = {}'.format(v.varName, v.x))

    print('Total Fleet Size:', (ni.sum(0, '*', '*') + uij.sum(0, '*', '*', '*') + cijk.sum(0, '*', '*', '*')).getValue())

    # Open a file for writing
    with open(f'../output/{output_path}.txt', 'w') as file:
        # Redirect standard output to the file
        old_stdout = sys.stdout
        sys.stdout = file
        print("results")
        # Check variable values and write them to the file
        for v in m.getVars():
            if v.x > 0:  # Print only non-zero variables for clarity
                print('{} = {}'.format(v.varName, v.x))

        # Restore standard output
        sys.stdout = old_stdout




