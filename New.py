import random
import cplex
from docplex.mp.model import Model
from datetime import datetime
import pandas as pd
from data import dfs, loads
from log import lg
import numpy as np


def planning(facility):
    start_time = datetime.now()
    random.seed(2)
    mdl = Model("Charging-cluster_Management")
    cpx = mdl.get_cplex()
    cpx.parameters.mip.pool.relgap.set(0.1)
    '''cpx.parameters.mip.tolerances.integrality.set(0)
    cpx.parameters.simplex.tolerances.markowitz.set(0.999)
    cpx.parameters.simplex.tolerances.optimality.set(1e-9)
    cpx.parameters.simplex.tolerances.feasibility.set(1e-9)
    cpx.parameters.mip.pool.intensity.set(2)
    cpx.parameters.mip.pool.absgap.set(1e75)
    cpx.parameters.mip.pool.relgap.set(1e75)
    cpx.parameters.mip.limits.populate.set(50)'''
    # Sets
    time = 1
    space_range = range(150)
    connector_range = range(4)
    vehicle_range = range(500)
    time_range = range(24)

    # Parameters
    S = 2000
    N = 6
    C_plug = 250 / 365 / 20
    C_EVSE = 4500 / 365 / 20
    C_grid = 240 / 365 / 20
    P_EVSE = 22 * time
    P_grid = 500
    n_s = 0.8
    l_star = 800
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
    cc = {}
    for i in connector_range:
        cc[i] = 1 + (i / 10000)
    price = {}

    for j in vehicle_range:
        for t in time_range:
            price[j, t] = 1 + (j + t / 10000)

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
    x = mdl.binary_var_dict(space_range, name='x')
    y = mdl.binary_var_matrix(connector_range, space_range, name='y')
    h = mdl.continuous_var_cube(space_range, vehicle_range, time_range, lb=0, name=f'h{j}')
    w = mdl.binary_var_matrix(space_range, vehicle_range, name='w')
    # e = mdl.continuous_var_cube(space_range, vehicle_range, time_range, lb=0, name='e')
    p_plus = mdl.continuous_var(lb=0, name='p_plus')
    p_star = mdl.continuous_var(lb=0, name='p_star')

    # Constraints

    mdl.add_constraint(mdl.sum(x[k] for k in space_range) <= S, 'C1')
    for k in space_range:
        mdl.add_constraint(mdl.sum(y[i, k] for i in connector_range) <= N * x[k], 'C3')

    for j in vehicle_range:
        mdl.add_constraint(mdl.sum(h[k, j, t] for k in space_range for t in range(A[j], D[j] + 1)) >=
                           n_s * e_d[j], 'C4')
    '''for j in vehicle_range:
        mdl.add_constraint(mdl.sum(h[k, j, t] for k in space_range for t in range(A[j], D[j] + 1))
                           <= e_d[j], 'C5')'''

    for t in time_range:
        mdl.add_constraint(mdl.sum(h[k, j, t] for k in space_range for j in vehicle_range) + l[t] <=
                           P_grid + p_plus, 'C6')

    for k in space_range:
        for t in time_range:
            mdl.add_constraint(mdl.sum(w[k, j] * U[j, t] for j in vehicle_range)
                               <= mdl.sum(y[i, k] for i in connector_range), 'C10')
    for j in vehicle_range:
        mdl.add_constraint(mdl.sum(w[k, j] for k in space_range) <= 1, 'C11')

    for k in space_range:
        for j in vehicle_range:
            for t in range(A[j], D[j] + 1):
                mdl.add_constraint(h[k, j, t] <= w[k, j] * P_EVSE, 'C15')

    for k in space_range:
        for t in time_range:
            mdl.add_constraint(mdl.sum(h[k, j, t] for j in vehicle_range) <= P_EVSE, 'C16')

    for t in time_range:
        mdl.add_constraint(mdl.sum(h[k, j, t] for k in space_range for j in vehicle_range) + l[t] - l_star
                           <= p_star, 'C17')

    mdl.minimize(mdl.sum(C_EVSE * c[k] * x[k] for k in space_range)
                 + mdl.sum(C_plug * cc[i] * y[i, k] for k in space_range for i in connector_range) + C_grid * p_plus
                 + mdl.sum(
        (T_e[t] * price[j, t]) * h[k, j, t] for k in space_range for j in vehicle_range for t in time_range)
                 + T_p * p_star)
    lg.error('Start_time: {}'.format(start_time))
    mdl.print_information()

    assert mdl.solve(), "!!! Solve of the model fails"
    lg.error(f'{mdl.report()}')
    lg.error(f'p_plus = {p_plus.solution_value}')
    lg.error(f'p_star = {p_star.solution_value}')
    for k in space_range:
        if x[k].solution_value != 0:
            lg.error(f'x_{k} = {x[k].solution_value}')

    for i in connector_range:
        for k in space_range:
            if y[i, k].solution_value != 0:
                lg.error(f'y_{i, k} = {y[i, k].solution_value}')

    for j in vehicle_range:
        for k in space_range:
            for t in time_range:
                if h[k, j, t].solution_value != 0:
                    lg.error(f'w_{k, j} = {w[k, j].solution_value}, '
                             f'h_{k, j, t} = {h[k, j, t].solution_value}, '
                             f'A_{j} = {A[j]}, D_{j} = {D[j]}, '
                             f'e_{j} = {e_d[j]}')
    v_index = pd.MultiIndex.from_product([space_range, vehicle_range, time_range],
                                         names=['charger', 'vehicle', 'time'])
    v_results = pd.DataFrame(-np.random.rand(len(v_index), 4), index=v_index)
    v_results.columns = ['Energy', 'Connection', 'Arrival', 'Departure']
    for j in vehicle_range:
        for k in space_range:
            for t in time_range:
                v_results.loc[(k, j, t), 'Connection'] = w[k, j].solution_value
                v_results.loc[(k, j, t), 'Energy'] = h[k, j, t].solution_value
                v_results.loc[(k, j, t), 'Arrival'] = A[j]
                v_results.loc[(k, j, t), 'Departure'] = D[j]
    v_results.to_csv(f'vehicles{facility}.csv')
    CS_index = pd.MultiIndex.from_product([space_range, connector_range],
                                          names=['charger', 'plug'])
    CS_results = pd.DataFrame(-np.random.rand(len(CS_index), 2), index=CS_index)
    CS_results.columns = ['CS', 'Connector']
    for k in space_range:
        for i in connector_range:
            CS_results.loc[(k, i), 'CS'] = x[k].solution_value
            CS_results.loc[(k, i), 'Connector'] = y[i, k].solution_value
    CS_results.to_csv(f'CS{facility}.csv')
    end_time = datetime.now()
    lg.error('Duration: {}'.format(end_time - start_time))


planning(facility=1)
