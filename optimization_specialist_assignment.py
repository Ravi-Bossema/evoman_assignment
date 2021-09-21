import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'evoman')
from demo_controller import player_controller

# imports other libs
import time
import numpy as np
from math import fabs,sqrt
import glob, os
from environment import Environment

# --- Constants ---
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"


experiment_name = 'individual_assignment'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

N_HIDDEN_NEURONS = 10
ENV = Environment(experiment_name=experiment_name,
                  enemies=[2],
                  playermode="ai",
                  player_controller=player_controller(N_HIDDEN_NEURONS),
                  enemymode="static",
                  level=2,
                  speed="fastest")
N_VARS = (ENV.get_num_sensors()+1)*N_HIDDEN_NEURONS + (N_HIDDEN_NEURONS+1)*5

DOM_L = -1
DOM_U = 1


class EA:
    def __init__(self, pop_size, std, generations, par, recom, mut, sur):
        self.pop_size = pop_size
        self.std = std
        self.generations = generations

        self.PSM = par
        self.RO = recom
        self.MO = mut
        self.SSM = sur

        # Initialize population
        self.population = np.random.uniform(DOM_L, DOM_U, (pop_size, N_VARS))

        self.f = self.evaluate(self.population)

        # Variables for plotting
        plot_f = self.f[self.f != np.inf]
        self.mean = [np.mean(plot_f)]
        self.upper = [np.amax(plot_f)]
        self.lower = [np.amin(plot_f)]

        # Best candidate to be tested at the end of the experiment
        self.best_candidate = None

    # Parent Selection
    def parent_selection(self):
        """Returns a list containing sets of two parents in accordance with the chosen parent selection mechanism"""
        if self.PSM == 'RS':
            return self.random_parent_selection()
        elif self.PSM == 'TS':
            return self.tournament_parent_selection()
        else:
            print("Please choose a parent_selection_mechanism of RS for random selection or TS for tournament selection")
            exit()

    def random_parent_selection(self):
        """Returns sets of two parents from the entire current population randomly without replacement"""
        parents = []
        population = list(self.population)
        for i in range(np.floor_divide(self.pop_size, 2)):
            i1, i2 = np.random.randint(0, len(population)-1, 2)
            couple = [population.pop(i1), population.pop(i2)]
            parents.append(couple)
        return parents

    def tournament_parent_selection(self):
        """Returns sets of parents from the entire current population using tournament selection with replacement"""
        parents = []
        for i in range(np.floor_divide(self.pop_size, 2)):
            k = min(10, self.pop_size)
            pool1, pool2 = np.random.randint(0, self.pop_size-1, (2, k))
            f1, f2 = -np.inf, -np.inf
            p1, p2 = None, None
            for x in range(k):
                if self.f[pool1[x]] > f1:
                    f1 = self.f[pool1[x]]
                    p1 = self.population[pool1[x]]
                if self.f[pool2[x]] > f2:
                    f2 = self.f[pool2[x]]
                    p2 = self.population[pool2[x]]
            couple = [p1, p2]
            parents.append(couple)
        return parents

    # Recombination
    def recombination(self, parents):
        """Returns a list of children from the parents in accordance with the chosen recombination operator"""
        if self.RO == 'UC':
            return self.uniform_crossover(parents)
        elif self.RO == 'PA':
            return self.partial_arithmetic(parents)
        else:
            print(
                "Please choose a recombination_operator of UC for uniform crossover or PA for partial arithmetic")
            exit()

    def uniform_crossover(self, parents):
        """Returns a list of children after recombining the parents using uniform crossover"""
        children = []
        for i in range(len(parents)):
            couple = parents[i]
            mask = np.random.choice([0, 1], N_VARS)
            child1 = mask*couple[0] + (1-mask)*couple[1]
            child2 = (1-mask)*couple[0] + mask*couple[1]
            children.append(child1)
            children.append(child2)
        return children

    def partial_arithmetic(self, parents):
        """Returns a list of children after recombining the parents using partial arithmetic"""
        children = []
        for i in range(len(parents)):
            couple = parents[i]
            mask = np.random.choice([0, 1], N_VARS)
            child1 = mask*couple[0] + (1-mask)/2*couple[0] + (1-mask)/2*couple[1]
            child2 = mask*couple[1] + (1-mask)/2*couple[0] + (1-mask)/2*couple[1]
            children.append(child1)
            children.append(child2)
        return children

    # Mutation
    def mutation(self, children):
        """Generates and returns a proposal population for the next generation
        based on the current generation's population pop and the chosen mutation operator"""
        if self.MO == 'RP':
            return self.random_perturbation(children)
        elif self.MO == 'DM':
            return self.differential_mutation(children)
        else:
            print("Please choose a mutation_operator of RP for random perturbation or DM for differential mutation")
            exit()

    def random_perturbation(self, children):
        """Generates a random mutation for each parameter of each individual and adds this mutation
        to said parameter.
        Returns the mutated population pop"""

        mutation_range = (DOM_U - DOM_L) * self.std
        for i in range(len(children)):
            ind = children[i]
            for param in range(N_VARS):
                # Since some domains, in particular the domain of ALPHA, are larger than others,
                # the mutation for those parameters should also be larger. Thus I set the mutation
                # range at std times the size of the domain
                mutation = np.random.normal(0, mutation_range)
                ind[param] = min(max(ind[param] + mutation, -1), 1)
            children[i] = ind
        return children

    def differential_mutation(self, children):
        """Generates a mutated population by sampling two different partners for each individual
        and adding the difference between said partners to the individual.
        Returns the mutated population pop"""
        for i in range(len(children)):
            ind = children[i]
            i1, i2 = np.random.randint(0, self.pop_size, 2)
            partner1 = self.population[i1]
            partner2 = self.population[i2]
            a = np.random.uniform(0.001, 2)
            mutation = a * (partner1 - partner2)
            for param in range(4):
                # To make sure we don't exit the domain we set the mutation to 0 if it would cause
                # the parameter to exit the domain
                if ind[param] + mutation[param] < DOM_L or \
                        ind[param] + mutation[param] > DOM_U:
                    mutation[param] = 0
            ind += mutation
            children[i] = ind
        return children

    # Survivor Selection
    def survivor_selection(self, children):
        """Selects and returns the population of the next generation in accordance with the
        chosen survival selection mechanism"""
        if self.SSM == 'GS':
            self.f = self.evaluate(children)
            return children
        elif self.SSM == 'FS':
            return self.fitness_based_selection(children)
        else:
            print("Please choose a survivor_selection_mechanism of GS for generational selection "
                  "or FS for fitness-based selection")
            exit()

    def fitness_based_selection(self, children):
        """Selects the best-performing 50% of the population and the offspring to continue into the next generation
        and randomly samples the remainder"""
        new_gen = np.empty([0, N_VARS])
        new_gen_f = np.empty(0)
        concat_pop = np.concatenate((self.population, children))
        concat_f = np.concatenate((self.f, self.evaluate(children)))
        order = np.flip(np.argsort(concat_f))
        for i in range(np.floor_divide(len(order), 2)):
            new_gen = np.append(new_gen, [concat_pop[order[i]]], axis=0)
            new_gen_f = np.append(new_gen_f, [concat_f[order[i]]], axis=0)
        rest = np.random.choice(order[np.floor_divide(len(order), 2):], self.pop_size-len(new_gen), replace=False)
        for i in rest:
            new_gen = np.append(new_gen, [concat_pop[i]], axis=0)
            new_gen_f = np.append(new_gen_f, [concat_f[i]], axis=0)
        self.f = new_gen_f
        return new_gen

    def evolve(self):
        """Runs the algorithm for each generation and returns the fitness of the final generation
        as well as plotting the relevant graphs"""
        for gen in range(self.generations):
            parents = self.parent_selection()
            children = self.recombination(parents)
            children = np.asarray(self.mutation(children))
            self.population = self.survivor_selection(children)
            print("Generation " + str(gen))  # To keep track of the process
            # For plotting
            plot_f = self.f[self.f != np.inf]
            self.mean.append(np.mean(plot_f))
            self.upper.append(np.amax(plot_f))
            self.lower.append((np.amin(plot_f)))

        self.plot_whole()
        self.plot_best()

        if self.SSM == 'GS':
            # Sort the population by the fitness and select the best performing one
            order = self.f.argsort()
            self.f = self.f[order[::-1]]
            self.population = self.population[order[::-1]]
            self.best_candidate = self.population[0]

        elif self.SSM == 'FS':
            # Using fitness based selection, the best individual is always selected first
            self.best_candidate = self.population[0]
            print(self.best_candidate)

        # Save the best solution of the last generation for testing
        f = open('individual_assignment/solutions_assignment/' + experiment_name + "_best_candidate.txt", 'w')
        np.savetxt('individual_assignment/solutions_assignment/' + experiment_name + "_best_candidate.txt",\
                   self.best_candidate)

    def simulation(self, env, x):
        f, p, e, t = env.play(pcont=x)
        return f

    def evaluate(self, x):
        return np.array(list(map(lambda y: self.simulation(ENV, y), x)))

    def plot_whole(self):
        """Plots the mean fitness and fitness range of each generation and saves it to a .png file"""
        t = np.arange(self.generations + 1)
        fig, ax = plt.subplots(1)
        ax.plot(t, self.mean, label='Mean fitness of the population', color='blue')
        ax.fill_between(t, self.lower, self.upper, label='Range of the population')
        ax.legend(loc='lower right')
        ax.set_xlabel('Generations')
        ax.set_ylabel('Fitness')
        ax.set_title('%s %s %s %s with a \u03C3 of %s' % (self.PSM, self.RO, self.MO, self.SSM, str(self.std)))
        ax.grid()
        plt.savefig('individual_assignment/' + experiment_name + '_plot_whole.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_best(self):
        """Plots the best-performing individual's fitness for each generation and saves it to a .png file"""
        t = np.arange(self.generations + 1)
        fig, ax = plt.subplots(1)
        ax.plot(t, self.upper, label='Best performing individual')
        ax.legend(loc='lower right')
        ax.set_xlabel('Generations')
        ax.set_ylabel('Fitness')
        ax.set_title('Best individuals of %s %s %s %s with a \u03C3 of %s' % (self.PSM, self.RO, self.MO, self.SSM, str(self.std)))
        ax.grid()
        plt.savefig('individual_assignment/' + experiment_name + '_plot_best.png', dpi=300, bbox_inches='tight')
        plt.show()


# Set hyperparameters
population_size = 5
generations = 1
standard_deviation = 0.1  # The factor with which the mutation range is determined
parent_selection_mechanism = 'RS'  # Either RS for random selection or TS for tournament selection
recombination_operator = 'PA'  # Either UC for uniform crossover or PA for partial arithmetic
mutation_operator = 'RP'  # Either RP for random perturbation or DM for differential mutation
survivor_selection_mechanism = 'FS'  # Either GS for generational selection or FS for fitness-based selection


ea = EA(population_size, standard_deviation, generations, parent_selection_mechanism, recombination_operator,
        mutation_operator, survivor_selection_mechanism)

ea.evolve()

"""#Have the best solution play against a selected enemy
for en in range(1, 9):
    ENV.update_parameter('enemies',[en])
    sol = np.loadtxt('individual_assignment/solutions_assignment/' + experiment_name + "_best_candidate.txt")
    ENV.play(sol)"""