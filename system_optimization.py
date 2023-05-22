from pulp import *
import pandas as pd
import numpy as np

# Constants
T = 1440 # For example, we can take time in hours of a day
K = 10 # For example, we can have 5 levels of SOC
V = [0, 1, 2] # Vertiports

# Flight time matrix
tau = [[0, 10, 0], [10, 0, 0], [0, 0, 0]]

# SOC levels drop matrix
kappa = [[0, 1, 0], [1, 0, 0], [0, 0, 0]]

gamma = [3.047619048, 3.047619048, 3.237872699, 3.729470167, 4.404786588, 5.379957014, 6.913363091,
        9.685271742, 16.30528373, 71.41103553]  # define gamma




# Create the 'prob' variable to contain the problem data
prob = LpProblem("Vertiport_Aircraft_Routing", LpMinimize)

# Decision variables
ni = LpVariable.dicts("n", [(t, i, k) for t in range(T+1) for i in V for k in range(K+1)], 0, None, LpInteger)
uij = LpVariable.dicts("u", [(t, i, j, k) for t in range(T+1) for i in V for j in V for k in range(K+1) if i != j], 0, None, LpInteger)



# initial
f_values = np.zeros((T, 2, 2))
data = pd.read_csv('schedule.csv')
LAX_DTLA = data[data['od'] == 'LAX_DTLA']
DTLA_LAX = data[data['od'] == 'DTLA_LAX']
# Create the list of lists
LAX_DTLA = [[1 if i in LAX_DTLA['schedule'].tolist() else 0] for i in range(1440)]
DTLA_LAX = [[1 if i in DTLA_LAX['schedule'].tolist() else 0] for i in range(1440)]
for t in range(T):
    f_values[t][0][1] = LAX_DTLA[t][0] # get the first (and only) item of the inner list
    f_values[t][1][0] = DTLA_LAX[t][0] # get the first (and only) item of the inner list

# # Make sure fi includes all valid (t, i, j) combinations
# fi = LpVariable.dicts("f", [(t, i, j) for t in range(T) for i in range(2) for j in range(2) if i != j], 0, None, LpInteger)

# # Then you can assign values like this:
# for t in range(T):
#     for i in range(2):
#         for j in range(2):
#             if i != j:  # Skip cases where i == j
#                 fi[(t, i, j)].setInitialValue(f_values[t][i][j])



# Objective function
prob += lpSum([ni[(0, i, k)] for i in V for k in range(K+1)])

# Constraints
for i in V:
    for k in range(K+1):
        for t in range(T+1):
            if i != 3:
                if t > 0:
                    prob += ni[(t, i, k)] == ni[(t-1, i, k)] + lpSum([uij[(t-1-tau[j-1][i-1], j, i, k+kappa[j-1][i-1])] for j in V if j != i and t-1-tau[j-1][i-1] >= 0 and (t-1-tau[j-1][i-1], j, i, k+kappa[j-1][i-1]) in uij]) - lpSum([uij[(t-1, i, j, k)] for j in V if j != i and (t-1, i, j, k) in uij])
                else:
                    prob += ni[(t, i, k)] == lpSum([uij[(t-1-tau[j-1][i-1], j, i, k+kappa[j-1][i-1])] for j in V if j != i and t-1-tau[j-1][i-1] >= 0 and (t-1-tau[j-1][i-1], j, i, k+kappa[j-1][i-1]) in uij]) - lpSum([uij[(t-1, i, j, k)] for j in V if j != i and (t-1, i, j, k) in uij])
            else:
                if k != 0:
                    if t > gamma[k-1]:
                        prob += ni[(t, i, k)] == ni[(t-1, i, k)] + lpSum([uij[(t-1-tau[j-1][i-1], j, i, k+kappa[j-1][i-1])] for j in V if j != i and t-1-tau[j-1][i-1] >= 0 and (t-1-tau[j-1][i-1], j, i, k+kappa[j-1][i-1]) in uij]) - lpSum([uij[(t-1, i, j, k)] for j in V if j != i and (t-1, i, j, k) in uij]) + ni[(t-gamma[k-1], i, k-1)] - ni[(t-gamma[k-1], i, k)]

for j in [0, 1]:
    for i in [0, 1]:
        if i != j:
            for t in range(T):
                prob += lpSum([uij[(t, i, j, k)] for k in range(K+1)]) >= f_values[t][i][j]

for i in V:
    for k in range(K+1):
        for t in range(T+1):
            if t > 0:
                prob += ni[(t-1, i, k)] >= lpSum([uij[(t-1, i, j, k)] for j in V if j != i])

for i in V:
    for k in range(K+1):
        prob += ni[(0, i, k)] == ni[(T, i, k)]

# Non-negativity constraints
for i in V:
    for j in V:
        if i != j:
            for k in range(K+1):
                for t in range(T+1):
                    prob += uij[(t, i, j, k)] >= 0

for i in V:
    for k in range(K+1):
        for t in range(T+1):
            prob += ni[(t, i, k)] >= 0

# Objective function
prob += lpSum([ni[(0, i, k)] for i in V for k in range(K+1)])

# The problem is solved using PuLP's choice of Solver
prob.solve()

# Each of the variables is printed with it's resolved optimum value
for v in prob.variables():
    print(v.name, "=", v.varValue)

# The optimized objective function value is printed to the console
print("Total Fleet Size = ", value(prob.objective))


