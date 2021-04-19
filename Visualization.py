import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from IPython.display import set_matplotlib_formats

set_matplotlib_formats('retina')
def visualize(i):

    vehicles = pd.read_csv(f'vehicles{i}.csv')

    # Energy Consumption
    dg = vehicles.groupby(['time']).agg({'Energy': 'sum'}).reset_index()
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8), constrained_layout=True)
    ax.plot(dg['time'], dg['Energy'],
            label='Energy', linestyle='--', marker='.')
    ax.set_xlabel("hour")
    ax.set_ylabel("Charging demand")
    ax.set_title("")
    ax.legend()

    # Utilization
    vehicles['Occupation'] = 0


    def clean(x, y, z, w):
        if x <= y <= z:
            return 1 * w
        else:
            return 0


    '''vehicles['Occupation'] = vehicles.apply(lambda row: clean(row['Arrival'], row['time'], row['Departure']
                                                              , row['Connection']), axis=1)'''
    dg = vehicles.groupby(['time']).agg({'Occupation': 'sum'}).reset_index()
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8), constrained_layout=True)
    ax.plot(dg['time'], dg['Occupation'],
            label='Occupation', linestyle='--', marker='.')
    ax.set_xlabel("hour")
    ax.set_ylabel("Charging demand")
    ax.set_title("")
    ax.legend()


    '''dg = vehicles.groupby(['time', 'charger']).agg({'Occupation': 'sum'}).reset_index()
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8), constrained_layout=True)
    for i in range(50):
        ax.plot(dg.loc[dg['charger'] == i]['time'], dg.loc[dg['charger'] == i]['Occupation'],
                label='Occupation', linestyle='--', marker='.')
    ax.set_xlabel("hour")
    ax.set_ylabel("Charging demand")
    ax.set_title("")'''

    '''dg1 = vehicles.groupby(['time', 'charger']).agg({'Energy': 'sum'})
    dgg = pd.DataFrame(dg1.stack()).reset_index()
    sns.lineplot(data=dgg, x='time', y='charger', ci=0)'''

    plt.show()
