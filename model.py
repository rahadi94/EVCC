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
    space_range = range(50)
    connector_range = range(4)
    vehicle_range = range(500)
    time_range = range(24)

    # Parameters
    SOC_0 = {}
    F = {}
    for j in vehicle_range:
        SOC_0[j] = random.randint(20, 70)
        F[j] = random.randint(SOC_0[j] + 10, 100)

    A = {}
    U = {}
    D = {}
    for j in vehicle_range:
        A[j] = random.randint(1, time_range[-1] - 2)
        D[j] = random.randint(A[j] + 1, time_range[-1] - 1)
        for t in time_range:
            if A[j] <= t <= D[j]:
                U[j, t] = 1
            else:
                U[j, t] = 0

    SOC_0 = {}
    F = {}
    for j in vehicle_range:
        SOC_0[j] = random.randint(20, 70)
        F[j] = min(random.randint(SOC_0[j] + 10, SOC_0[j] + 10 + (D[j] - A[j]) * 5), 100)

    # Variables
    x = mdl.binary_var_dict(space_range, name='x')
    y = mdl.binary_var_matrix(connector_range, space_range, name='y')
    h = mdl.continuous_var_matrix(space_range, time_range, name='h')
    z = mdl.binary_var_cube(space_range, vehicle_range, time_range, name='z')
    b = mdl.binary_var_cube(space_range, range(0, 5), time_range, name='b')
    w = mdl.binary_var_cube(space_range, vehicle_range, time_range, name='w')
    p = mdl.continuous_var_cube(space_range, vehicle_range, time_range, lb=0, name='p')
    SOC = {}
    P = 22

    # Constraints
    for j in vehicle_range:
        SOC[j] = mdl.integer_var_dict(range(A[j], D[j] + 1), lb=0, name=f'SOC_{j}')

    mdl.add_constraint(mdl.sum(x[k] for k in space_range) <= 300, 'C1')
    mdl.add_constraint(mdl.sum(y[i, k] for i in connector_range for k in space_range) <= 500, 'C2')
    for k in space_range:
        mdl.add_constraint(mdl.sum(y[i, k] for i in connector_range) <= 4 * x[k], 'C3')
    for j in vehicle_range:
        mdl.add_constraint(SOC[j][A[j]] == SOC_0[j], 'C4')

    for j in vehicle_range:
        for t in range(A[j] + 1, D[j] + 1):
            mdl.add_constraint(SOC[j][t] == SOC[j][t - 1] + mdl.sum(p[k, j, t] for k in space_range), 'C5')

    for j in vehicle_range:
        for k in space_range:
            for t in range(A[j] + 1, D[j] + 1):
                mdl.add_constraint(p[k, j, t] <= h[k, t], 'C6')

    for j in vehicle_range:
        for k in space_range:
            for t in range(A[j] + 1, D[j] + 1):
                mdl.add_constraint(p[k, j, t] <= z[k, j, t] * P, 'C20')

    # IF static power
    '''for j in vehicle_range:
        for t in range(A[j] + 1, D[j] + 1):
            mdl.add_constraint(SOC[j][t] == SOC[j][t - 1] + mdl.sum(10 * z[k, j, t] for k in space_range), 'C5')'''
    # IF SOC is a full matrix
    '''for j in vehicle_range:
        for t in range(D[j] + 1, time_range[-1] + 1):
            mdl.add_constraint(SOC[j][t] == SOC[j][t - 1], 'C6')'''

    for k in space_range:
        for t in time_range:
            mdl.add_constraint(mdl.sum(z[k, j, t] for j in vehicle_range) ==
                               mdl.sum(i * b[k, i, t] for i in range(0, 5)), 'C7')

    for k in space_range:
        for t in time_range:
            mdl.add_constraint(mdl.sum(b[k, i, t] for i in range(0, 5)) == 1, 'C8')

    for k in space_range:
        for t in time_range:
            mdl.add_constraint(h[k, t] <= P, 'C9')

    for k in space_range:
        for i in range(0, 5):
            for t in time_range:
                mdl.add_constraint((i * h[k, t] - 100 * (1 - b[k, i, t])) <= P, 'C10')

    for k in space_range:
        for j in vehicle_range:
            for t in time_range:
                mdl.add_constraint(w[k, j, t] <= x[k], 'C11')
    for k in space_range:
        for t in time_range:
            mdl.add_constraint(mdl.sum(w[k, j, t] for j in vehicle_range) <= mdl.sum(y[i, k] for i in connector_range)
                               , 'C7')
    for j in vehicle_range:
        for t in time_range:
            mdl.add_constraint(mdl.sum(w[k, j, t] for k in space_range) <= 1, 'C12')
    for k in space_range:
        for j in vehicle_range:
            for t in time_range:
                mdl.add_constraint(w[k, j, t] <= U[j, t], 'C13')
    for k in space_range:
        for j in vehicle_range:
            for t in time_range:
                mdl.add_constraint(z[k, j, t] <= w[k, j, t], 'C14')

    for j in vehicle_range:
        mdl.add_constraint(F[j] - SOC[j][D[j]] <= 20, 'C15')

    for k in space_range:
        for j in vehicle_range:
            for t in range(A[j] + 1, D[j] + 1):
                mdl.add_constraint(w[k, j, t] == w[k, j, t - 1], 'C16')

    mdl.minimize(mdl.sum(1000 * x[k] for k in space_range) + mdl.sum(100 * y[i, k] for k in space_range
                                                                     for i in connector_range))
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
                if w[k, j, t].solution_value != 0:
                    lg.error(f'w_{k, j, t} = {w[k, j, t].solution_value} , '
                             f'z_{k, j, t} = {z[k, j, t].solution_value}, A_{j}={A[j]}, D_{j}={D[j]}')

    for i in range(0, 5):
        for t in time_range:
            for k in space_range:
                if b[k, i, t].solution_value != 0:
                    lg.error(f'b_{k, i, t} = {b[k, i, t].solution_value}, '
                             f'z = {mdl.sum(z[k, j, t].solution_value for j in vehicle_range)}'
                             f', w = {mdl.sum(w[k, j, t].solution_value for j in vehicle_range)}')

    for t in time_range:
        for k in space_range:
            if h[k, t].solution_value != 0:
                lg.error(f'h_{k, t} = {h[k, t].solution_value}, '
                         f'sum = {mdl.sum(z[k, j, t].solution_value for j in vehicle_range)}')

    for j in vehicle_range:
        lg.error(f'SOC_0_{j} = {SOC_0[j]}, SOC_{j, D[j]} = {SOC[j][D[j]].solution_value}, F_{j}={F[j]}')

    # print(sum([F[j] - SOC[j][D[j]].solution_value]))
    end_time = datetime.now()
    lg.error('Duration: {}'.format(end_time - start_time))


planning()
