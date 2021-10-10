from optimization_generalist_assignment import EA

from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize

population_size = 50
generations = 10
PSM = 'TS'  # Either RS for random selection or TS for tournament selection

space = [Real(0.01, 0.2, name='std'),
         Integer(2, 20, name='survivor_select')]
if PSM == 'TS':
    space.append(Integer(3, 20, name='parent_k'))


@use_named_args(space)
def objective(**params):
    if PSM == 'TS':
        std, survivor_select, parent_k = params.values()
    else:
        std, survivor_select = params.values()
        parent_k = 10

    ea = EA(population_size, std, generations, PSM, survivor_select, parent_k)
    ea.evolve()

    return 100 - ea.upper[-1]   # The tuning function wants to minimize so we give it 100-fitness
                                # to get as close to 100 as possible


res_gp = gp_minimize(objective, space, n_calls=10)

print("Best score=%.4f" % res_gp.fun)
print("""Best parameters:
- std = %.6f
- survivor_select = %d""" % (res_gp.x[0], res_gp.x[1]))
if PSM == 'TS':
    print("- k = %d" % (res_gp.x[2]))
