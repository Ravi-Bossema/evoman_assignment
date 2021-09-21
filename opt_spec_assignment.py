
# imports framework
import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller
from numpy.random import randint
from numpy.random import rand

# imports other libs
import time
import numpy as np
from math import fabs,sqrt
import glob, os

import numpy as np
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import random as rnd

# imports other libs


# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"


experiment_name = 'Individual_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10

# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                  enemies=[2],
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest")

# default environment fitness is assumed for experiment

env.state_to_log()  # checks environment state


####   Optimization for controller solution (best genotype-weights for phenotype-network): Ganetic Algorihm    ###

ini = time.time()  # sets time marker


# genetic algorithm params

run_mode = 'train'  # train or test

# number of weights for multilayer with 10 hidden neurons
n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5


dom_u = 1
dom_l = -1
npop = 5
gens = 5
mutation = 0.2
last_best = 0
pop = np.random.uniform(dom_l, dom_u, (npop, n_vars))


# runs simulation
def simulation(env, x):
    f, p, e, t = env.play(pcont=x)
    return f



# evaluation


def evaluate(x):
    return np.array(list(map(lambda y: simulation(env, y), x)))


def selection(population):
    '''Selects an individual randomly based on its fitness value as weights'''
    # Calculate the fitness values of every individual in the population
    fitness_values = evaluate(pop)
    # Calculate the relative fitness
    max_fitness = [max(fitness_values) for x in fitness_values]
    # Assign weights to each individual based on fitness
    weights = [x / sum(max_fitness) for x in max_fitness]
    # Select a individual as a parent with weighted randomness
    selected_individual = rnd.choices(pop, weights)[0]

    return selected_individual


def crossover(parent1, parent2, prnt=False):
    '''Returns the child after crossover between the parents'''
    # Select cut points
    c1 = rnd.randint(0, len(parent1) - 2)
    c2 = rnd.randint(c1 + 1, len(parent2) - 1)

    # Create an Empty Child DNA
    child = ["X"] * len(parent1)

    # Set the values between the cut points from parent1 in the child DNA
    child[c1:c2 + 1] = parent1[c1:c2 + 1]
    if prnt: print('Child DNA from Parent 1', child)

    # Fill the remaining values from parent2
    for i in range(len(child)):
        for j in range(len(parent2)):
            # If the parent value is not already in the child then
            if parent2[j] not in child:
                # Replace with parent value only at places marked X
                child[i] = parent2[j] if child[i] == "X" else child[i]
                # break out of the inner loop and move over to the next position in the DNA
                break

    if prnt: print('Child DNA after adding Parent 2', child)

    return child


def mutation(individual):
    '''Mutates the DNA of a child/individual by swapping the values at two positions'''
    # Selecting the index values to swap
    pos_1 = rnd.randint(0, len(individual) - 1)
    pos_2 = rnd.randint(0, len(individual) - 1)
    # Init the mutant
    mutant = individual.copy()
    # Swap
    mutant[pos_1] = individual[pos_2]
    mutant[pos_2] = individual[pos_1]

    return mutant

# loads file with the best solution for testing
if run_mode =='test':

    bsol = np.loadtxt(experiment_name+'/best.txt')
    print( '\n RUNNING SAVED BEST SOLUTION \n')
    env.update_parameter('speed','normal')
    evaluate([bsol])

    sys.exit(0)


# initializes population loading old solutions or generating new ones

if not os.path.exists(experiment_name+'/evoman_solstate'):

    print( '\nNEW EVOLUTION\n')

    pop = np.random.uniform(dom_l, dom_u, (npop, n_vars))
    fit_pop = evaluate(pop)
    print(fit_pop)
    best = max(fit_pop)
    mean = int(np.mean(fit_pop))
    std = int(np.std(fit_pop))
    best_fit= np.where(fit_pop == best)
    best_fit_ind = fit_pop[best_fit]
    ini_g = 0
    solutions = [pop, fit_pop]
    env.update_solutions(solutions)


else:

    print( '\nCONTINUING EVOLUTION\n')

    env.load_state()
    pop = env.solutions[0]
    fit_pop = env.solutions[1]

    best = max(fit_pop)
    mean = np.mean(fit_pop)
    std = np.std(fit_pop)

    # finds last generation number
    file_aux  = open(experiment_name+'/gen.txt','r')
    ini_g = int(file_aux.readline())
    file_aux.close()

# saves results for first pop
file_aux  = open(experiment_name+'/results.txt','a')
file_aux.write('\n\ngen best mean std')
print( '\n GENERATION '+str(ini_g)+' '+str(int(best_fit_ind))+' '+str(mean)+' '+str(std))
file_aux.write('\n'+str(ini_g)+' '+str(int(best_fit_ind))+' '+str(mean)+' '+str(std))
file_aux.close()



def optimize_GA(pop_size, max_generations, crossover_prob, mutate_prob):
    '''Returns the final solution by optimizing using genetic algorithm'''
    global_best = {}

    # Start Evolution
    for g in range(max_generations):

        # Calculate Fitness of the population
        pop = np.random.uniform(dom_l, dom_u, (npop, n_vars))
        population_fitness = evaluate(pop)
        # Get the individual with the best fitness value
        best_fitness = max(population_fitness)
        print(best_fitness)
        print('best fit printed')
        print(population_fitness)
        best_fit_index= np.where(population_fitness == best_fitness)
        best_fit_individual = pop[best_fit_index]

        # Check with global best
        if g == 0:
            global_best['fitness'] = best_fitness
            global_best['dna'] = best_fit_individual
        else:
            if best_fitness >= global_best['fitness']:
                global_best['fitness'] = best_fitness
                global_best['dna'] = best_fit_individual
                print('Best Solution at Generation', g)

        new_population = []
        for i in range(pop_size):
            # Select the parents
            parent1 = selection(pop)
            parent2 = selection(pop)

            # Crossover between the parents with a certain probability
            if rnd.random() <= crossover_prob:
                child = crossover(parent1, parent2)
            else: # or directly clone one of the parents
                child = rnd.choice([parent1, parent2])

            # Mutation
            if rnd.random() <= mutate_prob:
                child = mutation(child)

            # Add child to new population
            new_population.append(child)
        best_fit = np.where(population_fitness == best_fitness)
        best_fit_individual = population_fitness[best_fit]
        std = int(np.std(population_fitness))
        mean = int(np.mean(population_fitness))
        # saves results
        file_aux = open(experiment_name + '/results.txt', 'a')
        print('\n GENERATION ' + str(g) + ' ' + str(int(best_fit_individual)) + ' ' + str(mean) + ' ' + str(std))
        file_aux.write('\n' + str(g) + ' ' + str(int(best_fit_individual)) + ' ' + str(mean) + ' ' + str(std))
        file_aux.close()
        # saves results
        file_plot = open('plot.txt', 'a')
        file_plot.write('\n' + str(g) + ' ' + str(int(best_fit_individual)) + ' ' + str(mean) + ' ' + str(std))
        file_plot.close()
    return global_best

pop_size = 10
max_generations = 10
crossover_prob = 0.95
mutate_prob = 0.7



# Start Optimization with Genetic Algorithm
output = optimize_GA(pop_size, max_generations, crossover_prob, mutate_prob)
print(output)

fim = time.time() # prints total execution time for experiment
print( '\nExecution time: '+str(round((fim-ini)/60))+' minutes \n')


env.state_to_log() # checks environment state

data = np.loadtxt('plot.txt')
x = data[:, 0]
y = data[:, 1]
plt.plot(x, y, color= 'blue', marker= 'o')
plt.title('Best Fitness Vs Generation')
plt.show()

x2 = data[:, 0]
y2 = data[:, 2]
plt.plot(x2, y2)
plt.title('Mean Fitness Vs Generation')
plt.show()

x3 = data[:, 0]
y3 = data[:, 3]
plt.plot(x3, y3)
plt.title('STD Vs Generation')
plt.show()

