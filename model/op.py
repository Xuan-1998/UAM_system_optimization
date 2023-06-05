from gurobipy import *
import pandas as pd
import numpy as np



def number_aircrafts_lp(tau, kappa, gamma, schedule, schedule_time_step, output_path, cost_ratio_flight2plane = 0):
    # Flight time matrix
    tau = np.array(tau) / 5
    tau = np.ceil(tau)

    # SOC levels drop matrix
    kappa = np.array(kappa) / 10
    kappa = np.ceil(kappa)

    # Charging Time matrix
    gamma = np.array(gamma) / 5
    gamma = np.ceil(gamma)

    max_flight_time = int(np.max(tau))

    T = schedule_time_step + 1 + max_flight_time

    # Constants
    K = 8
    V = [0, 1]

    f_values = np.zeros((T, 2, 2))
    data = pd.read_csv(f'../input/{schedule}.csv')
    LAX_DTLA = data[data['od'] == 'LAX_DTLA']
    DTLA_LAX = data[data['od'] == 'DTLA_LAX']

    # Create the list of lists
    LAX_DTLA = [[1 if i in LAX_DTLA['schedule'].tolist() else 0] for i in range(1440)]
    DTLA_LAX = [[1 if i in DTLA_LAX['schedule'].tolist() else 0] for i in range(1440)]

    # Reshape the array
    new_array_LAX_DTLA = np.array(LAX_DTLA).reshape((288, 5)).tolist()
    new_array_DTLA_LAX = np.array(DTLA_LAX).reshape((288, 5)).tolist()

    # Add elements within each cell
    new_array_LAX_DTLA_sum = np.sum(new_array_LAX_DTLA, axis=1)
    new_array_DTLA_LAX_sum = np.sum(new_array_DTLA_LAX, axis=1)

    LAX_DTLA = new_array_LAX_DTLA_sum # Binary flight schedule for 5-mins intervals
    DTLA_LAX = new_array_DTLA_LAX_sum

    for t in range(T-max_flight_time-1):
        f_values[t+1][0][1] = LAX_DTLA[t] # get the first (and only) item of the inner list
        f_values[t+1][1][0] = DTLA_LAX[t] # get the first (and only) item of the inner list

    variables = {f'a{str(i)}': [] for i in range(2, int(sum(gamma)) + 1)}

    def find_combinations(k):
        a = []
        for i in range(len(gamma)):
            for j in range(len(gamma)):
                if i != j:
                    if sum(gamma[i:j+1]) == k:
                        a.append([i,j])
                if gamma[i] == k:
                    a.append([i])
        a_unique = list(set(tuple(i) for i in a))
        a_unique = [item[0] if len(item) == 1 else item for item in a_unique]
        return a_unique

    for i in range(2, int(sum(gamma)) + 1):
        variables['a' + str(i)] = find_combinations(i)

    # Create a new model
    m = Model("Vertiport_Aircraft_Routing")

    # Create variables
    ni = m.addVars(((t, i, k) for t in range(T) for i in V for k in range(K+1)), vtype=GRB.INTEGER, name="n")
    uij = m.addVars(((t, i, j, k) for t in range(T) for i in V for j in V for k in range(K+1) if i != j), vtype=GRB.INTEGER, name="u")
    cijk = m.addVars(((t, i, x, y) for t in range(T) for i in V for x in range(K+1) for y in range(K+1) if x < y), vtype=GRB.INTEGER, name="c")

    m.setObjective(ni.sum(0, '*', '*') + uij.sum(0, '*', '*', '*') + cijk.sum(0, '*', '*', '*')+ cost_ratio_flight2plane*uij.sum('*', '*', '*', '*'), GRB.MINIMIZE)

    # Dynamic equation
    for i in V:
        for k in range(K+1):
            for t in range(1, T):
                m.addConstr(
                    ni[t, i, k] == ni[t-1, i, k] + 
                    quicksum(uij[t-tau[j][i], j, i, k+kappa[j][i]] for j in V if j != i and t-1-tau[j][i] >= 0 and k+kappa[j][i] <= K) -
                    quicksum(uij[t, i, j, k] for j in V if j != i) +
                    quicksum(cijk[t-sum(gamma[x:k]), i, x, k] for x in range(k) if t-sum(gamma[x:k]) >= 0) -
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

    print('Total Fleet Size:', m.objVal)

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




