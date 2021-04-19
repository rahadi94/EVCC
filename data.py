import pandas as pd
import numpy as np
import random

random.seed(2)
df = pd.read_csv('Parking+Charging_Data_BLENDED_CLUSTERED.csv')

df1 = df[df['SiteID'] == 'Facility_1'][df['EntryDate'] == '2019-01-10'][df['EntryDate']
                                                                        == df['ExitDate']][df['HoursStay'] >= 1]
dfs = df1.sample(n=500, weights=df.groupby('ClusterNum')['ClusterNum'].transform('count'))

loads = pd.read_csv('Scaled_Avg_Hourly_Load.csv')
loads = loads.loc[(loads['SiteID'] == 'Facility_1')].groupby('hour').agg({'kWScaled': 'mean'})
