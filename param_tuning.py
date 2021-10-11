from optimization_generalist_assignment import EA

from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize

param_control = False

n = 0
space = [Real(0.01, 0.5, name='std'),
         Integer(2, 20, name='survivor_select'),
         Integer(3, 20, name='parent_k'),
         Integer(4, 10, name='n_generations')]
if param_control:
    space.append(Real(-0.05, 0.05, name='mutation_step_size'))


@use_named_args(space)
def objective(**params):
    mut_step = 0
    if param_control:
        std, survivor_select, parent_k, n_generations, mut_step = params.values()
    else:
        std, survivor_select, parent_k, n_generations = params.values()

    population_size = 220 / (n_generations+1)

    print('-------------------- TUNING RUN ' + str(n) + ' --------------------')

    ea = EA(population_size, std, n_generations, survivor_select, parent_k, mut_step)
    ea.evolve()

    return 100 - ea.upper[-1]   # The tuning function wants to minimize so we give it 100-fitness
                                # to get as close to 100 as possible


res_gp = gp_minimize(objective, space, n_calls=30)

print("Best score=%.4f" % res_gp.fun)
print("""Best parameters:
- std = %.6f
- survivor_select = %d
- k = %d
- n_generations = %d""" % (res_gp.x[0], res_gp.x[1], res_gp.x[2], res_gp.x[3]))
if param_control:
    print('- mutation_step_size = %.6f' % res_gp.x[4])
