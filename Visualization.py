import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from IPython.display import set_matplotlib_formats

set_matplotlib_formats('retina')

vehicles = pd.read_csv('vehicles1.csv')

'''dg = vehicles.groupby(['time', 'charger']).agg({'Energy': 'sum'}).reset_index()
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8), constrained_layout=True)
for i in range(50):
    ax.plot(dg.loc[dg['charger'] == i]['time'], dg.loc[dg['charger'] == i]['Energy'],
            label='Energy', linestyle='--', marker='.')
ax.set_xlabel("hour")
ax.set_ylabel("Charging demand")
ax.set_title("")'''
dg = vehicles.groupby(['time']).agg({'Energy': 'sum'}).reset_index()
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8), constrained_layout=True)
ax.plot(dg['time'], dg['Energy'],
        label='Energy', linestyle='--', marker='.')
ax.set_xlabel("hour")
ax.set_ylabel("Charging demand")
ax.set_title("")
ax.legend()

'''dg1 = vehicles.groupby(['time', 'charger']).agg({'Energy': 'sum'})
dgg = pd.DataFrame(dg1.stack()).reset_index()
sns.lineplot(data=dgg, x='time', y='charger', ci=0)'''

plt.show()
