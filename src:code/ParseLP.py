import pandas as pd
import numpy as np
import ast
import seaborn as sns
import matplotlib.pyplot as plt

def convert2df(output_file):
    results = pd.read_table(f'../output/{output_file}.txt')
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

    return specificn, specificu, specificc


def calculate_num_aircrafts(specificc, specificu, specificn, gamma, flight_time=np.array([[0,2],[2,0]])):
    end = 288 + 1 + int(np.max(flight_time))
    all_c = np.zeros(shape=(1,end), dtype=int)
    for i in range(specificc.shape[0]):
        val = int(specificc['amount'][i])
        soc0 = int(specificc['x'][i])
        soc1 = int(specificc['y'][i])
        t = int(specificc['t'][i])
        time_charge = int(gamma[soc0: soc1].sum())
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
            flight[j][t:t+flight_time[origin, dest]] = 1

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

    return all_c, all_n, all_u
