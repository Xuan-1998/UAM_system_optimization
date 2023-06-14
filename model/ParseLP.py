import pandas as pd
import ast
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

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


def calculate_num_aircrafts(specificc, 
                            specificu, 
                            specificn, 
                            gamma = np.array([0.0129,0.0133,0.0137,0.0142,0.0147,
                                          0.0153,0.0158,0.0166,0.0172,0.018,
                                          0.0188,0.0197,0.0207,0.0219,0.0231,
                                          0.0245,0.026,0.0278,0.03,0.0323,
                                          0.0351,0.0384,0.0423,0.0472,0.0536,
                                          0.0617,0.0726,0.0887,0.1136,0.1582,
                                          0.2622,0.9278,])*60, 
                            flight_time=np.array([[0,2],[2,0]])):
    end = 288 + 1 + int(np.max(flight_time))
    all_c = np.zeros(shape=(1,end), dtype=int)
    for i in range(specificc.shape[0]):
        val = int(specificc['amount'][i])
        soc0 = int(specificc['x'][i])
        soc1 = int(specificc['y'][i])
        t = int(specificc['t'][i])
        time_charge = int(np.ceil(gamma[soc0: soc1].sum()/5))
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

def plot_charging_policy(specificc, name, gamma = np.array([0.0129,0.0133,0.0137,0.0142,0.0147,
                                          0.0153,0.0158,0.0166,0.0172,0.018,
                                          0.0188,0.0197,0.0207,0.0219,0.0231,
                                          0.0245,0.026,0.0278,0.03,0.0323,
                                          0.0351,0.0384,0.0423,0.0472,0.0536,
                                          0.0617,0.0726,0.0887,0.1136,0.1582,
                                          0.2622,0.9278,])*60):
    occupied = np.zeros(shape=(1, gamma.shape[0] + 1), dtype=int)
    all_c01 = np.zeros(shape=(1,gamma.shape[0] + 1), dtype=int)
    for k in range(specificc['t'].max() + 1):
        for i in range(specificc.shape[0]):
            val = int(specificc['amount'][i])
            soc0 = int(specificc['x'][i])
            soc1 = int(specificc['y'][i])
            t = int(specificc['t'][i])
            if t == k:
                for j in range(soc0, soc1 + 1):
                    occupied[0][j] = occupied[0][j] + val
        all_c01 = np.concatenate([all_c01, occupied], axis=0)
        occupied = np.zeros(shape=(1, gamma.shape[0] + 1), dtype=int)
    all_c01 = all_c01[1:,:]

    # Assuming your data is stored in a NumPy array called 'data'
    data = all_c01
    dataT = data.T
    temp = np.zeros(shape=(data.shape[1], data.shape[0]))
    for i in range(dataT.shape[0]):
        temp[dataT.shape[0] - i - 1] = dataT[i]

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Create the heatmap
    heatmap = ax.imshow(temp, cmap='hot', aspect='auto')

    # Set the x-axis labels
    ax.set_xticks(np.arange(0, 291, 30))  # Adjust the step size as needed
    ax.set_xticklabels(np.arange(0, 291, 30))

    # Set the y-axis labels
    ax.set_yticks(np.arange(0, 33))
    ax.set_yticklabels(np.arange(33, 0, -1))  # Reverse the order of y-axis labels

    # Add a colorbar
    cbar = plt.colorbar(heatmap)

    # Set the title and labels
    ax.set_title('Heatmap of Time Steps and SOC Levels for Charging Policy ' + name)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('SOC Level')

    # Display the plot
    plt.show()

def plot_SOC_demand_time(specificn, demand, name):
    specificn['k'] = specificn['k'] * specificn['amount']
    specificn = specificn.groupby('t').sum().reset_index()
    fig, ax1 = plt.subplots()

    # Plotting Total SOC of the system
    ax1.plot(specificn['t'], specificn['k'], label='Total SOC of the system', color='blue')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Total SOC')
    ax1.tick_params(axis='y')

    # Creating a twin Axes object
    ax2 = ax1.twinx()

    # Plotting Number of Passengers
    ax2.plot(demand['passenger_arrival_time'], demand['passenger_id'], label='Number of Passengers', color='red')
    ax2.set_ylabel('Number of Passengers')
    ax2.tick_params(axis='y')

    # Combining the legends from both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()

    # Assigning colors to the legends
    legend1 = ax1.legend(lines_1, labels_1, loc='upper left', frameon=False)
    for text in legend1.get_texts():
        text.set_color('blue')

    legend2 = ax2.legend(lines_2, labels_2, loc='upper right', frameon=False)
    for text in legend2.get_texts():
        text.set_color('red')
    plt.title('System’s SOCs vs Demand over time for Charging Policy ' + name)
    plt.show()

