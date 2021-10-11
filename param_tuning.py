from optimization_generalist_assignment import EA

from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize

n = 0
space = [Real(0.01, 0.2, name='std'),
         Integer(2, 20, name='survivor_select'),
         Integer(3, 20, name='parent_k'),
         Integer(4, 10, name='n_generations')]


@use_named_args(space)
def objective(**params):
    std, survivor_select, parent_k, n_generations = params.values()

    population_size = 220 / (n_generations+1)

    print('-------------------- TUNING RUN ' + str(n) + ' --------------------')

    ea = EA(population_size, std, n_generations, survivor_select, parent_k)
    ea.evolve()

    return 100 - ea.upper[-1]   # The tuning function wants to minimize so we give it 100-fitness
                                # to get as close to 100 as possible


res_gp = gp_minimize(objective, space, n_calls=30)

print("Best score=%.4f" % res_gp.fun)
print("""Best parameters:
- std = %.6f
- survivor_select = %d
- k = %d""" % (res_gp.x[0], res_gp.x[1], res_gp.x[2]))
