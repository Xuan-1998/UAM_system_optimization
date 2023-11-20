from gurobipy import *
import pandas as pd
import numpy as np
import os
import sys
import time

def time_limit_callback(model, where):
    if where == GRB.Callback.MIP:
        # Check the runtime
        runtime = model.cbGet(GRB.Callback.RUNTIME)
        # If 15 minutes have passed, stop the optimization
        if runtime > 900: # 900 seconds = 15 minutes
            model.terminate()


def spill_op(flight_schedule,
            occupancy,
            schedule_time_step,
            fleet_size,
            output_path,
            seat_capacity=4,
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

    ########################################Work on this part##############################################
    pij = np.zeros((T, 2, 2))
    bins = np.arange(0, 24*60+1, 5)
    merged = pd.concat([flight_schedule, occupancy['num_pax']], axis=1)
    merged['time_bins'] = pd.cut(merged['schedule'], bins, right=False)
    pax_count = merged.groupby(['od', 'time_bins'])['num_pax'].sum().unstack('od', fill_value=0)
    LAX_DTLA = np.array(pax_count['LAX_DTLA'].tolist())
    DTLA_LAX = np.array(pax_count['DTLA_LAX'].tolist())

    for t in range(T-max_flight_time-1):
        pij[t+1][0][1] = LAX_DTLA[t] # get the first (and only) item of the inner list
        pij[t+1][1][0] = DTLA_LAX[t] # get the first (and only) item of the inner list
    ########################################Work on this part##############################################
    
    
    # Create a new model
    m = Model("Spill Optimal Policy")
    # m.setParam('OutputFlag', 0)
    m.setParam('threads', 2)
    m.setParam('Method', 2)
    m.setParam('MIPGap', 0.05)
    m.setParam('FeasibilityTol', 1e-7)


    # Create variables
    ni = m.addVars(((t, i, k) for t in range(T) for i in V for k in range(K+1)), vtype=GRB.INTEGER, name="n")
    uij = m.addVars(((t, i, j, k) for t in range(T) for i in V for j in V for k in range(K+1) if i != j), vtype=GRB.INTEGER, name="u")
    cijk = m.addVars(((t, i, x, y) for t in range(T) for i in V for x in range(K+1) for y in range(K+1) if x < y), vtype=GRB.INTEGER, name="c")
    sij = m.addVars(((t, i, j) for t in range(T) for i in V for j in V  if i!=j), vtype=GRB.INTEGER, name="s")



    # m.setObjective(fixed_cost*sij.sum('*', '*', '*') + 
    #                variable_cost*(uij.sum('*', '*', '*', '*')), GRB.MINIMIZE)
    m.setObjective(fixed_cost*sij.sum('*', '*', '*'), GRB.MINIMIZE)
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
                    
    # for j in V:
    #     for i in V:
    #         if i != j:
    #             for t in range(T-1):
    #                 m.addConstr(uij.sum(t, i, j, '*') >= f_values[t][i][j])

    # Can't fly when SOC = 0
    for i in V:
        for j in V:
            for t in range(T):
                if i != j:
                    m.addConstr(uij[t, i, j, 0] == 0)

    # Constraint on spill
    for i in V:
        for j in V:
            for t in range(T):
                if i != j:
                    m.addConstr(sij[t, i, j] >= 0)
                    m.addConstr(sij[t, i, j] >= pij[t, i, j] - seat_capacity*uij.sum(t, i, j, '*'))

    # Fleet size constraint
    m.addConstr(ni.sum(0, '*', '*') + uij.sum(0, '*', '*', '*') + cijk.sum(0, '*', '*', '*') == fleet_size)


    

    # Integrate new variables
    m.update()

    # Solve model
    # m.optimize(time_limit_callback)
    start_time = time.time()
    relax = m.relax()
    relax.optimize()
    end_time = time.time()
    print('Time elapsed:', end_time - start_time)

    # Calculate the total fleet size and store it in a variable
    total_spill = sij.sum('*', '*', '*').getValue() 

    # Print the total fleet size
    print('Total Spill:', total_spill)

    # If you want to save it to a file, you can do something like this:
    with open(output_path+'_total_spill.txt', 'w') as file:
        file.write('Total Spill: ' + str(total_spill))

    with open(output_path+'_spill_op_result.txt', 'w') as file:
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

