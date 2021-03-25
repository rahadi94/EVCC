import random
import cplex
from docplex.mp.model import Model
from datetime import datetime
from log import lg


def planning():
    start_time = datetime.now()
    random.seed(2)
    mdl = Model("Charging-cluster_Management")

    # Sets
    space_range = range(20)
    connector_range = range(4)
    vehicle_range = range(100)
    time_range = range(24)

    # Parameters
    S = 50
    N = 4
    C_plug = 100
    C_EVSE = 1000
    C_grid = 10
    P_EVSE = 22
    P_grid = 10000
    n_s = 0.9
    l_star = 150

    T_e = {}
    l = {}
    for t in time_range:
        T_e[t] = random.randint(25, 80) / 100
        l[t] = 100

    e_d = {}
    A = {}
    U = {}
    D = {}
    for j in vehicle_range:
        e_d[j] = random.randint(20, 40)
        A[j] = random.randint(1, time_range[-1] - 2)
        D[j] = random.randint(A[j] + 1, time_range[-1] - 1)
        for t in time_range:
            if A[j] <= t <= D[j]:
                U[j, t] = 1
            else:
                U[j, t] = 0

    # Variables
    x = mdl.binary_var_dict(space_range, name='x')
    y = mdl.binary_var_matrix(connector_range, space_range, name='y')
    h = mdl.continuous_var_cube(space_range, vehicle_range, time_range, name='h')
    w = mdl.binary_var_cube(space_range, vehicle_range, time_range, name='w')
    e = mdl.continuous_var_cube(space_range, vehicle_range, time_range, lb=0, name='e')
    p_plus = mdl.continuous_var(ub=0, name='p_plus')
    p_star = mdl.continuous_var(ub=0, name='p_star')

    # Constraints

    mdl.add_constraint(mdl.sum(x[k] for k in space_range) <= S, 'C1')
    # mdl.add_constraint(mdl.sum(y[i, k] for i in connector_range for k in space_range) <= S * N, 'C2')
    for k in space_range:
        mdl.add_constraint(mdl.sum(y[i, k] for i in connector_range) <= N * x[k], 'C3')

    mdl.add_constraint(mdl.sum(h[k, j, t] for k in space_range for j in vehicle_range for t in time_range)
                       >= n_s * mdl.sum(e_d[j] for j in vehicle_range), 'C4')
    for j in vehicle_range:
        mdl.add_constraint(mdl.sum(h[k, j, t] for k in space_range for t in time_range)
                           <= e_d[j], 'C5')

    for t in time_range:
        mdl.add_constraint(mdl.sum(h[k, j, t] for k in space_range for j in vehicle_range) + l[t] <=
                           P_grid + p_plus, 'C6')

    '''for k in space_range:
        for j in vehicle_range:
            mdl.add_constraint(w[k, j, A[j]] <= 0, 'C7')
            mdl.add_constraint(w[k, j, D[j]] <= 0, 'C8')'''

    for k in space_range:
        for j in vehicle_range:
            for t in range(A[j] + 1, D[j] + 1):
                mdl.add_constraint(w[k, j, t] <= x[k], 'C9')
    for k in space_range:
        for t in time_range:
            mdl.add_constraint(mdl.sum(w[k, j, t] for j in vehicle_range) <= mdl.sum(y[i, k] for i in connector_range)
                               , 'C10')
    for j in vehicle_range:
        for t in range(A[j], D[j] + 1):
            mdl.add_constraint(mdl.sum(w[k, j, t] for k in space_range) <= 1, 'C11')
    for k in space_range:
        for j in vehicle_range:
            for t in time_range:
                mdl.add_constraint(w[k, j, t] <= U[j, t], 'C12')

    for k in space_range:
        for j in vehicle_range:
            for t in range(A[j] + 1, D[j] + 1):
                mdl.add_constraint(w[k, j, t] >= w[k, j, t - 1], 'C13')
                mdl.add_constraint(w[k, j, t] <= w[k, j, t - 1], 'C13')
    for k in space_range:
        for j in vehicle_range:
            for t in time_range:
                mdl.add_constraint(h[k, j, t] <= w[k, j, t] * P_EVSE, 'C14')

    for k in space_range:
        for t in time_range:
            mdl.add_constraint(mdl.sum(h[k, j, t] for j in vehicle_range) <= P_EVSE, 'C15')

    mdl.minimize(mdl.sum(C_EVSE * x[k] for k in space_range) + mdl.sum(C_plug * y[i, k] for k in space_range
                                                                       for i in connector_range) + C_grid * p_plus +
                 mdl.sum(T_e[t] * h[k, j, t] for k in space_range for j in vehicle_range for t in time_range))
    # + mdl.sum(F[j] - SOC[j][D[j]] for j in vehicle_range) * 1 )

    mdl.print_information()

    # assert mdl.solve(), "!!! Solve of the model fails"
    mdl.solve()
    mdl.report()
    for k in space_range:
        if x[k].solution_value != 0:
            lg.error(f'x_{k} = {x[k].solution_value}')
    for i in connector_range:
        for k in space_range:
            if y[i, k].solution_value != 0:
                lg.error(f'y_{i, k} = {y[i, k].solution_value}')

    for j in vehicle_range:
        for t in time_range:
            for k in space_range:
                if w[k, j, t].solution_value != 0 or h[k, j, t].solution_value != 0:
                    lg.error(f'w_{k, j, t} = {w[k, j, t].solution_value}, '
                             f'h_{k, j, t} = {h[k, j, t].solution_value}, '
                             f'A_{j} = {A[j]}, D_{j} = {D[j]}')

    end_time = datetime.now()
    lg.error('Duration: {}'.format(end_time - start_time))


planning()
