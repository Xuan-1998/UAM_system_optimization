from gurobipy import *
import pandas as pd
import numpy as np

# Constants
T = 1440
K = 10
V = [0, 1, 2]

# Flight time matrix
tau = [[0, 6.06667, 0], [6.06667, 0, 0], [0, 0, 0]]

# SOC levels drop matrix
kappa = [[0, 1, 0], [1, 0, 0], [0, 0, 0]]

gamma = [3.047619048, 3.047619048, 3.237872699, 3.729470167, 4.404786588, 5.379957014, 6.913363091,
        9.685271742, 16.30528373, 71.41103553]

# Example 'schedule.csv' data loading (Please replace with actual file loading if needed)
# Here, it is simply initialized with zeros.
# initial
f_values = [[[0, 0], [0, 0]] for _ in range(T)]
data = pd.read_csv('schedule.csv')
LAX_DTLA = data[data['od'] == 'LAX_DTLA']
DTLA_LAX = data[data['od'] == 'DTLA_LAX']
# Create the list of lists
LAX_DTLA = [[1 if i in LAX_DTLA['schedule'].tolist() else 0] for i in range(1440)]
DTLA_LAX = [[1 if i in DTLA_LAX['schedule'].tolist() else 0] for i in range(1440)]
for t in range(T):
    f_values[t][0][1] = LAX_DTLA[t][0] # get the first (and only) item of the inner list
    f_values[t][1][0] = DTLA_LAX[t][0] # get the first (and only) item of the inner list



# Create a new model
m = Model("Vertiport_Aircraft_Routing")


# Create variables
ni = m.addVars(((t, i, k) for t in range(T+1) for i in V for k in range(K+1)), vtype=GRB.INTEGER, name="n")
uij = m.addVars(((t, i, j, k) for t in range(T+1) for i in V for j in V for k in range(K+1) if i != j), vtype=GRB.INTEGER, name="u")

# Define the objective
m.setObjective(ni.sum(0, '*', '*'), GRB.MINIMIZE)

# Constraints
# Please note that because of Gurobi's zero-based indexing, you might need to adjust the indices
# Here, the indices are not adjusted
for i in V:
    for k in range(K+1):
        for t in range(T+1):
            if i != 3:
                if t > 0:
                    m.addConstr(ni[t, i, k] == ni[t-1, i, k] + quicksum(uij[t-1-tau[j-1][i-1], j, i, k+kappa[j-1][i-1]] for j in V if j != i and t-1-tau[j-1][i-1] >= 0 and (t-1-tau[j-1][i-1], j, i, k+kappa[j-1][i-1]) in uij) - quicksum(uij[t-1, i, j, k] for j in V if j != i and (t-1, i, j, k) in uij))
                else:
                    m.addConstr(ni[t, i, k] == quicksum(uij[t-1-tau[j-1][i-1], j, i, k+kappa[j-1][i-1]] for j in V if j != i and t-1-tau[j-1][i-1] >= 0 and (t-1-tau[j-1][i-1], j, i, k+kappa[j-1][i-1]) in uij) - quicksum(uij[t-1, i, j, k] for j in V if j != i and (t-1, i, j, k) in uij))

# The rest of your constraints
for i in V:
    for k in range(K+1):
        if i != 3:
            for t in range(T+1):
                if k != 0:
                    if t > gamma[k-1]:
                        m.addConstr(ni[t, i, k] == ni[t-1, i, k] + quicksum(uij[t-1-tau[j-1][i-1], j, i, k+kappa[j-1][i-1]] for j in V if j != i and t-1-tau[j-1][i-1] >= 0 and (t-1-tau[j-1][i-1], j, i, k+kappa[j-1][i-1]) in uij) - quicksum(uij[t-1, i, j, k] for j in V if j != i and (t-1, i, j, k) in uij) + ni[int(t-gamma[k-1]), i, k-1] - ni[int(t-gamma[k-1]), i, k])

for j in [0, 1]:
    for i in [0, 1]:
        if i != j:
            for t in range(T):
                m.addConstr(uij.sum(t, i, j, '*') >= f_values[t][i][j])

for i in V:
    for k in range(K+1):
        for t in range(T+1):
            if t > 0:
                m.addConstr(ni[t-1, i, k] >= uij.sum(t-1, i, '*', k))

for i in V:
    for k in range(K+1):
        m.addConstr(ni[0, i, k] == ni[T, i, k])

# Integrate new variables
m.update()

# Solve model
m.optimize()

if m.status == GRB.Status.INFEASIBLE:
    print('The model is infeasible; computing IIS')
    m.computeIIS()
    m.write("model.ilp")
    m.feasRelaxS(0, False, False, True) # calculate relaxed solution
    m.optimize()



# Print results
for v in m.getVars():
    if v.x > 0:  # Print only non-zero variables for clarity
        print('{} = {}'.format(v.varName, v.x))

print('Total Fleet Size:', m.objVal)
