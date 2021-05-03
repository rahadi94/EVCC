import random
from docplex.mp.model import Model
from datetime import datetime
import pandas as pd

from data import data_preparation
from log import lg
import numpy as np


def planning(facility, data):
    start_time = datetime.now()
    random.seed(2)
    mdl = Model("Charging-cluster_Management")
    # Sets
    dfs, loads = data_preparation(facility, data)
    time = 1
    space_range = range(25)
    vehicle_range = range(dfs['EntryHour'].count())
    time_range = range(24)

    # Parameters
    S = 2000
    N = 6
    C_plug = 250 / 365 / 2
    C_EVSE = 4500 / 365 / 20 + 250 / 365 / 2
    C_grid = 240 / 365 / 2
    P_EVSE = 22 * time
    P_grid = loads.max().values[0] * 1.5
    n_s = 1
    l_star = loads.max().values[0]
    T_p = 15.48 / 30
    T_e = {}
    '''T = [8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 9, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6
        , 23, 23.1, 23.2, 23.3, 23.4, 23.5, 7.9, 8]'''
    if time == 1:
        T = [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8
            , 23, 23, 23, 23, 23, 23, 8, 8]
    else:
        T = [8, 8, 8, 8, 8, 8, 8
            , 23, 23, 23, 8, 8]
    '''T = [27.10, 27.00, 25.58, 25.02, 25.27, 28.19, 32.32, 39.45, 41.82, 40.70, 40.60,
     40.59, 39.08, 39.85, 39.45, 45.58, 50.84, 54.26, 58.40, 49.07, 43.55, 38.25, 38.10, 34.44]'''
    l = {}
    for t in time_range:
        T_e[t] = T[t] / 100
        # T_e[t] = 0.25
        l[t] = loads.iloc[t, 0]
    c = {}
    for k in space_range:
        c[k] = 1 + (k / 10000)
    price = {}
    for j in vehicle_range:
        for t in time_range:
            price[j, t] = 1 # + (j + t / 10000)

    e_d = {}
    A = {}
    U = {}
    D = {}
    for j in vehicle_range:
        A[j] = int(dfs.iloc[j]['EntryHour'] / time)
        D[j] = int(dfs.iloc[j]['ExitHour'] / time)
        e_d[j] = float(dfs.iloc[j]['final_kWhRequested'])
        for t in time_range:
            if A[j] <= t <= D[j]:
                U[j, t] = 1
            else:
                U[j, t] = 0

    # Variables
    # x = mdl.binary_var_dict(space_range, name='x')
    # h = mdl.continuous_var_cube(space_range, vehicle_range, time_range, lb=0, name=f'h{j}')
    w = mdl.binary_var_matrix(space_range, vehicle_range, name='w')
    e = mdl.continuous_var_matrix(vehicle_range, time_range, lb=0, name='e')
    l = mdl.continuous_var_dict(vehicle_range, lb=0, name='l')
    p_plus = mdl.continuous_var(lb=0, name='p_plus')
    p_star = mdl.continuous_var(lb=0, name='p_star')

    # Constraints

    for j in vehicle_range:
        mdl.add_constraint( n_s * e_d[j] - mdl.sum(e[j, t] for t in range(A[j], D[j] + 1)) <= l[j], 'C4')

    for t in time_range:
        mdl.add_constraint(mdl.sum(e[j, t] for j in vehicle_range) + l[t] <=
                           P_grid + p_plus, 'C6')

    for t in time_range:
        mdl.add_constraint(mdl.sum(w[k, j] * U[j, t] for k in space_range for j in vehicle_range)
                           <= 25, 'C10')
    for j in vehicle_range:
        mdl.add_constraint(mdl.sum(w[k, j] for k in space_range) <= 1, 'C11')

    for j in vehicle_range:
        for t in range(A[j], D[j] + 1):
            mdl.add_constraint(e[j, t] <= mdl.sum(w[k, j] for k in space_range) * P_EVSE, 'C15')

    for t in time_range:
        mdl.add_constraint(mdl.sum(e[j, t] for j in vehicle_range) + l[t] - l_star
                           <= p_star, 'C17')

    c1 = C_grid * p_plus
    c2 = mdl.sum(
        (T_e[t]) * e[j, t] for j in vehicle_range for t in time_range) + T_p * p_star
    l = mdl.sum(l[j] for j in vehicle_range)
    mdl.minimize(c1 + c2 + l)
    lg.error('Start_time: {}'.format(start_time))
    mdl.print_information()

    assert mdl.solve(), "!!! Solve of the model fails"
    lg.error(f'{mdl.report()}')
    lg.error(f'p_plus = {p_plus.solution_value}')
    lg.error(f'p_star = {p_star.solution_value}')
    lg.error('c1: {}'.format(c1.solution_value))
    lg.error('c2: {}'.format(c2.solution_value))

    for j in vehicle_range:
        for k in space_range:
            for t in time_range:
                if e[j, t].solution_value * w[k, j].solution_value != 0:
                    lg.error(f'w_{k, j} = {w[k, j].solution_value}, '
                             f'e_{j, t} = {e[j,t].solution_value}, '
                             f'A_{j} = {A[j]}, D_{j} = {D[j]}, '
                             f'e_{j} = {e_d[j]}')
    v_index = pd.MultiIndex.from_product([vehicle_range, time_range],
                                         names=['vehicle', 'time'])
    v_results = pd.DataFrame(-np.random.rand(len(v_index), 5), index=v_index)
    v_results.columns = ['Energy', 'Connection', 'Arrival', 'Departure', 'Occupation']
    for j in vehicle_range:
        for t in time_range:
            # v_results.loc[(k, j, t), 'Connection'] = w[k, j].solution_value
            v_results.loc[(j, t), 'Energy'] = e[j, t].solution_value
            v_results.loc[(j, t), 'Arrival'] = A[j]
            v_results.loc[(j, t), 'Departure'] = D[j]
            v_results.loc[(j, t), 'Occupation'] = U[j, t]
    v_results.to_csv(f'vehicles{facility}.csv')

    end_time = datetime.now()
    lg.error('Duration: {}'.format(end_time - start_time))


dates = ['2019-06-03', '2019-06-04', '2019-06-05', '2019-06-06', '2019-06-07',
         '2019-10-21', '2019-10-22', '2019-10-23', '2019-10-24', '2019-10-25']

facilities = ['Facility_3', 'Facility_4', 'Facility_6']
'''for j in ['2019-06-05', '2019-06-06', '2019-06-07',
         '2019-10-21', '2019-10-22', '2019-10-23', '2019-10-24', '2019-10-25']:'''
'''dates = ['2019-06-06', '2019-06-07',
         '2019-10-21', '2019-10-22', '2019-10-23', '2019-10-24', '2019-10-25']'''
#for i in dates:
planning(facility='Facility_1', data='2019-06-03')