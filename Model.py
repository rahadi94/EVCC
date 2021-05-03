import random
from docplex.mp.model import Model
from datetime import datetime
import pandas as pd
from data import data_preparation
from log import lg
import numpy as np


def planning(facility, date, plug=4, adoption=1, service_level=1):
    start_time = datetime.now()
    random.seed(2)
    mdl = Model("Charging-cluster_Management")
    mdl.parameters.timelimit = 7200 * 2
    # mdl.parameters.threads = 32
    # mdl.parameters.simplex.tolerances.optimality = 1e-1
    # mdl.parameters.feasopt.tolerance = 1e-1
    # mdl.parameters.mip.strategy.lbheur = 1
    # mdl.parameters.optimalitytarget = 2
    # mdl.parameters.benders.strategy = 3
    # mdl.parameters.mip.strategy.bbinterval = 1
    # mdl.parameters.mip.tolerances.absmipgap = 0.2
    mdl.parameters.mip.tolerances.mipgap = 0.01
    # mdl.parameters.mip.limits.nodes = 100

    # Sets
    time = 1
    dfs, loads = data_preparation(facility, date)
    Event = dfs['EntryHour'].count()
    dfs = dfs.sample(n=int(Event*adoption))
    space_range = range(80)
    connector_range = range(plug)
    vehicle_range = range(dfs['EntryHour'].count())
    lg.error(f'Number of Events={dfs["EntryHour"].count()}')
    time_range = range(5, 23)

    # Parameters
    S = 2000
    N = 6
    C_plug = 250 / 365 * 0.5
    C_EVSE = 4500 / 365 * 0.5
    C_grid = 240 / 365 * 0.5
    P_EVSE = 22 * time
    P_grid = loads.max().values[0] * 1.5
    n_s = service_level
    l_star = loads.max().values[0]
    T_p = 15.48 / 30
    T_e = {}
    if date in ['2019-06-03', '2019-06-04', '2019-06-05', '2019-06-06', '2019-06-07']:
        T = [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8
            , 23, 23, 23, 23, 23, 23, 8, 8]
    else:
        '''T = [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8
            , 23, 23, 23, 23, 23, 23, 8, 8]'''
        T = [8, 8, 8, 8, 8, 8, 8, 8, 8, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5
            , 23, 23, 23, 23, 23, 23, 8, 8]

    l = {}
    for t in time_range:
        T_e[t] = T[t] / 100
        # T_e[t] = 0.25
        # l[t] = loads.iloc[t, 0]
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
            price[j, t] = 1 # + ((j + t) / 10000)

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

    c1 = mdl.sum(C_EVSE * c[k] * x[k] for k in space_range) \
         + mdl.sum(C_plug * cc[i] * y[i, k] for k in space_range for i in connector_range) + C_grid * p_plus
    c2 = mdl.sum(
        (T_e[t]) * h[k, j, t] for k in space_range for j in vehicle_range for t in time_range) + T_p * p_star
    mdl.minimize(c1 + c2)
    lg.error('Start_time: {}'.format(start_time))
    mdl.print_information()
    '''import warnings
    warnings.filterwarnings("error")'''
    try:
        assert mdl.solve(), "!!! Solve of the model fails"
        end_time = datetime.now()
        lg.error('Duration: {}'.format(end_time - start_time))
        NoC = mdl.sum(x[k] for k in space_range)
        NoP = mdl.sum(y[i, k] for k in space_range for i in connector_range)
        lg.error('c1: {}'.format(c1.solution_value))
        lg.error('c2: {}'.format(c2.solution_value))
        lg.error('NoC: {}'.format(NoC.solution_value))
        lg.error('NoP: {}'.format(NoP.solution_value))

        # lg.error("it did not solved")
        mdl.report()
        lg.error(f'facility = {facility}, date = {date}')
        lg.error(f'objective_function = {mdl.objective_value}')
        lg.error(f'p_plus = {p_plus.solution_value}')
        lg.error(f'p_star = {p_star.solution_value}')
        '''for k in space_range:
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
                                 f'e_{j} = {e_d[j]}')'''
        v_index = pd.MultiIndex.from_product([space_range, vehicle_range, range(24)],
                                             names=['charger', 'vehicle', 'time'])
        v_results = pd.DataFrame(-np.random.rand(len(v_index), 5), index=v_index)
        v_results.columns = ['Energy', 'Connection', 'Arrival', 'Departure', 'Occupation']
        for j in vehicle_range:
            for k in space_range:
                for t in time_range:
                    v_results.loc[(k, j, t), 'Connection'] = w[k, j].solution_value
                    v_results.loc[(k, j, t), 'Energy'] = h[k, j, t].solution_value
                    v_results.loc[(k, j, t), 'Arrival'] = A[j]
                    v_results.loc[(k, j, t), 'Departure'] = D[j]
                    v_results.loc[(k, j, t), 'Occupation'] = w[k, j].solution_value * U[j, t]
        v_results.to_csv(f'vehicles_{facility}_{date}.csv')
        CS_index = pd.MultiIndex.from_product([space_range, connector_range],
                                              names=['charger', 'plug'])
        CS_results = pd.DataFrame(-np.random.rand(len(CS_index), 2), index=CS_index)
        CS_results.columns = ['CS', 'Connector']
        for k in space_range:
            for i in connector_range:
                CS_results.loc[(k, i), 'CS'] = x[k].solution_value
                CS_results.loc[(k, i), 'Connector'] = y[i, k].solution_value
        CS_results.to_csv(f'CS_{facility}_{date}.csv')

        results = pd.read_csv('results_service.csv')

        results.loc[(results['Service_Level'] == service_level) & (results['Date'] == date), 'Installation_costs'] = c1.solution_value
        results.loc[(results['Service_Level'] == service_level) & (results['Date'] == date), 'Operations_costs'] = c2.solution_value
        results.loc[(results['Service_Level'] == service_level) & (results['Date'] == date), 'NoC'] = NoC.solution_value
        results.loc[(results['Service_Level'] == service_level) & (results['Date'] == date), 'NoP'] = NoP.solution_value
        results.loc[(results['Service_Level'] == service_level) & (results['Date'] == date), 'P_plus'] = p_plus.solution_value
        results.loc[(results['Service_Level'] == service_level) & (results['Date'] == date), 'P_star'] = p_star.solution_value
        results.loc[(results['Service_Level'] == service_level) & (results['Date'] == date), 'costs'] = mdl.objective_value
        results.loc[(results['Service_Level'] == service_level) & (results['Date'] == date), 'Event'] = dfs["EntryHour"].count()
        results.to_csv('results_service.csv')

    except:
        lg.error('it did not solve')
        lg.error(f'facility = {facility}, date = {date}')
        end_time = datetime.now()
        lg.error('Duration: {}'.format(end_time - start_time))
        return


'''dates = ['2019-06-03', '2019-06-04', '2019-06-05', '2019-06-06', '2019-06-07',
         '2019-10-21', '2019-10-22', '2019-10-23', '2019-10-24', '2019-10-25']'''
dates = ['2019-06-03', '2019-06-04', '2019-06-05', '2019-06-06', '2019-06-07']

for i in dates:
    planning(facility='Facility_3', date=i, plug=4, adoption=1, service_level=0.6)
