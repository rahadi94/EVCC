import pandas as pd
import numpy as np
import random

random.seed(2)


def data_preparation(facility, date):
    loads = pd.read_csv('Scaled_Avg_Hourly_Load.csv')
    loads = loads.loc[(loads['SiteID'] == 'Facility_KoeBogen')].groupby('hour').agg({'kWScaled': 'mean'})

    df = pd.read_csv('Parking+Charging_Data_BLENDED_CLUSTERED.csv')

    df1 = df[(df['SiteID'] == facility) & (df['EntryDate'] == date)
             & (df['EntryDate'] == df['ExitDate']) &
             (df['HoursStay'] >= 1) & (df['EntryHour'] > 4) & (df['ExitHour'] <= 22)]
    number = 450
    while True:
        dfs = df1.sample(n=number)
        occupation = pd.DataFrame(range(24))
        for i in range(24):
            occupation.iloc[i][0] = dfs[(dfs['EntryHour'] <= i) & (dfs['ExitHour'] >= i)]['EntryHour'].count()
        if occupation[0].max() <= 250:

            return dfs, loads
        number -= 1
        # print(number)
    # dfs = df1.sample(n=500, weights=df.groupby('ClusterNum')['ClusterNum'].transform('count'))
