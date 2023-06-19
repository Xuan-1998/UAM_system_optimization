from gurobipy import *
import pandas as pd
import numpy as np



def number_aircrafts_lp(schedule, 
                        schedule_time_step,
                        output_path,
                        tau=[[0, 10], [10, 0]], 
                        kappa = [[0, 10], [10, 0]], 
                        gamma = np.array([0.0129,0.0133,0.0137,0.0142,0.0147,
                                          0.0153,0.0158,0.0166,0.0172,0.018,
                                          0.0188,0.0197,0.0207,0.0219,0.0231,
                                          0.0245,0.026,0.0278,0.03,0.0323,
                                          0.0351,0.0384,0.0423,0.0472,0.0536,
                                          0.0617,0.0726,0.0887,0.1136,0.1582,
                                          0.2622,0.9278,])*60, 
                        fixed_cost=1, 
                        variable_cost=0.00001):
    
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
                   variable_cost*(uij.sum('*', '*', '*', '*')), GRB.MINIMIZE)

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
        
        

def number_aircrafts_lp_DTLA_charging(schedule, 
                        schedule_time_step,
                        output_path,
                        tau=[[0, 5.92], [5.85, 0]], 
                        kappa = [[0, 7.71875], [7.44375, 0]], 
                        gamma = np.array([0.0129,0.0133,0.0137,0.0142,0.0147,
                                          0.0153,0.0158,0.0166,0.0172,0.018,
                                          0.0188,0.0197,0.0207,0.0219,0.0231,
                                          0.0245,0.026,0.0278,0.03,0.0323,
                                          0.0351,0.0384,0.0423,0.0472,0.0536,
                                          0.0617,0.0726,0.0887,0.1136,0.1582,
                                          0.2622,0.9278,])*60, 
                        fixed_cost=1, 
                        variable_cost=0.01):
    
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
                   variable_cost*(uij.sum('*', '*', '*', '*')), GRB.MINIMIZE)

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
    # Only Charge at DTLA
    m.addConstr(cijk.sum('*', 0, '*', '*') == 0)
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
    # m.optimize()
    # if m.status == GRB.Status.INFEASIBLE:
    #     print('The model is infeasible; computing IIS')
    #     m.computeIIS()
    #     m.write("model.ilp")
    #     m.feasRelaxS(0, False, False, True) # calculate relaxed solution
    #     m.optimize()

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



def number_aircrafts_lp_middle_charging(schedule, 
                        schedule_time_step,
                        output_path,
                        tau=[[0, 5.92, 3.25], [5.85, 0, 3.85], [3.25, 3.85, 0]], 
                        kappa = [[0, 7.71875, 4.85625], [7.44375, 0, 6.45], [4.85625, 6.45, 0]], 
                        gamma = np.array([0.0129,0.0133,0.0137,0.0142,0.0147,
                                          0.0153,0.0158,0.0166,0.0172,0.018,
                                          0.0188,0.0197,0.0207,0.0219,0.0231,
                                          0.0245,0.026,0.0278,0.03,0.0323,
                                          0.0351,0.0384,0.0423,0.0472,0.0536,
                                          0.0617,0.0726,0.0887,0.1136,0.1582,
                                          0.2622,0.9278,])*60, 
                        fixed_cost=1, 
                        variable_cost=0.00000001):
    
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
    V1 = [0, 1, 2]

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
    ni = m.addVars(((t, i, k) for t in range(T) for i in V1 for k in range(K+1)), vtype=GRB.INTEGER, name="n")
    uij = m.addVars(((t, i, j, k) for t in range(T) for i in V1 for j in V1 for k in range(K+1) if i != j), vtype=GRB.INTEGER, name="u")
    cijk = m.addVars(((t, i, x, y) for t in range(T) for i in V1 for x in range(K+1) for y in range(K+1) if x < y), vtype=GRB.INTEGER, name="c")

    m.setObjective(fixed_cost*(ni.sum(0, '*', '*') + 
                   uij.sum(0, '*', '*', '*') + 
                   cijk.sum(0, '*', '*', '*')) + 
                   variable_cost*(uij.sum('*', '*', '*', '*')), GRB.MINIMIZE)

    # Dynamic equation
    for i in V1:
        for k in range(K+1):
            for t in range(1, T):
                m.addConstr(
                    ni[t, i, k] == ni[t-1, i, k] + 
                    quicksum(uij[t-tau[j][i], j, i, k+kappa[j][i]] for j in V1 if j != i and t-1-tau[j][i] >= 0 and k+kappa[j][i] <= K) -
                    quicksum(uij[t, i, j, k] for j in V1 if j != i) +
                    quicksum(cijk[t-np.ceil(sum(gamma[x:k])/5), i, x, k] for x in range(k) if t-np.ceil(sum(gamma[x:k])/5) >= 0) -
                    quicksum(cijk[t, i, k, y] for y in range(k+1, K+1))
                )

    # Stationary Constraint
    for k in range(K+1):
        for i in V1:
            for j in V1:
                if i != j:
                    m.addConstr(ni[0, i, k] == ni[schedule_time_step+max_flight_time, i, k])
                    m.addConstr(uij[0, i, j, k] == uij[schedule_time_step+max_flight_time, i, j, k])
            
    for x in range(K+1):
        for y in range(K+1):
            for i in V1:
                if (x < y):
                    m.addConstr(cijk[0, i, x, y] == cijk[schedule_time_step+max_flight_time, i, x, y])
                    
    for j in V:
        for i in V:
            if i != j:
                for t in range(T-1):
                    m.addConstr(uij.sum(t, i, j, '*') >= f_values[t][i][j])

    # Can't fly when SOC = 0
    for i in V1:
        for j in V1:
            for t in range(T):
                if i != j:
                    m.addConstr(uij[t, i, j, 0] == 0)
                    
    m.addConstr(cijk.sum('*', 1, '*', '*') == 0)
    m.addConstr(cijk.sum('*', 0, '*', '*') == 0)

    # Integrate new variables
    m.update()

    m.Params.MIPGap = 0.13  # Set the optimality tolerance to 5%
    m.Params.FeasibilityTol = 1e-7
    # Solve model
    m.optimize()
    # if m.status == GRB.Status.INFEASIBLE:
    #     print('The model is infeasible; computing IIS')
    # m.computeIIS()
    # m.write("model.ilp")
    # m.feasRelaxS(0, False, False, True) # calculate relaxed solution
    # m.optimize()

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