import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'evoman')
from demo_controller import player_controller

# imports other libs
import time
import statistics
import numpy as np
from math import fabs,sqrt
import glob, os
from environment import Environment

# --- Constants ---
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"


experiment_name = 'generalist_assignment_TS'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

N_HIDDEN_NEURONS = 10
ENEMIES = [6,7,8]

ENV = Environment(experiment_name=experiment_name,
                  enemies=ENEMIES,
                  multiplemode="yes",
                  playermode="ai",
                  player_controller=player_controller(N_HIDDEN_NEURONS),
                  enemymode="static",
                  level=2,
                  speed="fastest")
N_VARS = (ENV.get_num_sensors()+1)*N_HIDDEN_NEURONS + (N_HIDDEN_NEURONS+1)*5

DOM_L = -1
DOM_U = 1


class EA:
    def __init__(self, pop_size, std, generations, survivor_percentage, parent_k, mut_step):
        self.pop_size = pop_size
        self.generations = generations
        self.std = std
        self.survivor_selection_percentage = survivor_percentage
        self.tuned_k = parent_k
        self.mutation_step_size = mut_step

        # Initialize population
        self.population = np.random.uniform(DOM_L, DOM_U, (pop_size, N_VARS))

        self.f = self.evaluate(self.population)

        # Variables for plotting
        self.mean = [np.mean(self.f)]
        self.upper = [np.amax(self.f)]
        self.lower = [np.amin(self.f)]

        # Best candidate to be tested at the end of the experiment
        self.best_candidate = None

    # Parent Selection
    def parent_selection(self):
        """Returns sets of parents from the entire current population using tournament selection with replacement"""
        parents = []
        for i in range(np.floor_divide(self.pop_size, 2)):
            k = min(self.tuned_k, self.pop_size)
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

    def recombination(self, parents):
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

    def mutation(self, children):
        """Generates a random mutation for each parameter of each individual and adds this mutation
        to said parameter.
        Returns the mutated population pop"""

        for i in range(len(children)):
            ind = children[i]
            for param in range(N_VARS):
                # Since some domains, in particular the domain of ALPHA, are larger than others,
                # the mutation for those parameters should also be larger. Thus I set the mutation
                # range at std times the size of the domain
                mutation = np.random.normal(0, self.std)
                ind[param] = min(max(ind[param] + mutation, -1), 1)
            children[i] = ind
        return children

    def survivor_selection(self, children):
        """Selects the best-performing 25% of the population and the offspring to continue into the next generation
        and randomly samples the remainder"""
        new_gen = np.empty([0, N_VARS])
        new_gen_f = np.empty(0)
        concat_pop = np.concatenate((self.population, children))
        concat_f = np.concatenate((self.f, self.evaluate(children)))
        order = np.flip(np.argsort(concat_f))
        for i in range(np.floor_divide(len(order), self.survivor_selection_percentage)):
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

            self.std += self.mutation_step_size     # For parameter control
            self.std = max(0.01, self.std)

        # Save the best solution of the last generation for testing
        if not os.path.exists(experiment_name + '/solutions_enemy' + str(ENEMIES)):
            os.makedirs(experiment_name + '/solutions_enemy' + str(ENEMIES))
        np.savetxt(experiment_name + '/solutions_enemy' + str(ENEMIES) + '/best_candidate_' + str(experiment)
                   + '.txt', self.population[0])
        np.savetxt(experiment_name + '/solutions_enemy' + str(ENEMIES) + '/mean_list_' + str(experiment)
                   + '.txt', self.mean)
        np.savetxt(experiment_name + '/solutions_enemy' + str(ENEMIES) + '/upper_list_' + str(experiment)
                   + '.txt', self.upper)

    def simulation(self, env, x):
        f, p, e, t = env.play(pcont=x)
        return f

    def evaluate(self, x):
        return np.array(list(map(lambda y: self.simulation(ENV, y), x)))


if __name__ == '__main__':
    def plot_whole(gen, m, m_std, u, u_std):
        """Plots the mean fitness and fitness range of each generation and saves it to a .png file"""
        t = np.arange(gen + 1)
        fig, ax = plt.subplots(1)
        ax.plot(t, m, label='Mean')
        ax.fill_between(t, m+m_std, m-m_std, facecolor='lightskyblue')
        ax.plot(t, u, label='Maximum')
        ax.fill_between(t, u+u_std, u-u_std, facecolor='moccasin')
        ax.legend(loc='lower right')
        ax.set_xlabel('Generations')
        ax.set_ylabel('Fitness')
        ax.set_xticks(np.arange(0, generations + 1, 1))
        ax.set_yticks(np.arange(0, 110, 10))
        ax.set_title('Average fitness over ' + str(n_exp) + ' experiments for enemy ' + str(ENEMIES))
        ax.grid()
        plt.savefig(experiment_name + '/plot_enemy' + str(ENEMIES) + '.png',
                    dpi=300, bbox_inches='tight')
        plt.show()


    # Set hyperparameters
    population_size = 94
    generations = 18
    n_exp = 10
    # Hyperparameters that are to be tuned with Sequential Parameter Optimization
    standard_deviation = 0.091309  # The factor with which the mutation range is determined
    mut_step = -0.001534
    survivor_selection_percentage = 13  # Population is divided by this value --> 4 will lead to 25% selected
    parent_selection_k = 12  # Default value is 10

    mean_list = []
    upper_list = []

    for experiment in range(n_exp):
        #ea = EA(population_size, standard_deviation, generations, survivor_selection_percentage, parent_selection_k, mut_step)
        #ea.evolve()
        mean = np.loadtxt(experiment_name + '/solutions_enemy' + str(ENEMIES) + '/mean_list_' + str(experiment)
                   + '.txt')
        mean_list.append(mean)
        upper = np.loadtxt(experiment_name + '/solutions_enemy' + str(ENEMIES) + '/upper_list_' + str(experiment)
                          + '.txt')
        upper_list.append(upper)

    mean_std = np.std(mean_list, axis=0)
    mean = np.mean(mean_list, axis=0)
    upper_std = np.std(upper_list, axis=0)
    upper = np.mean(upper_list, axis=0)

    plot_whole(generations, mean, mean_std, upper, upper_std)

    """To generate the boxplots for evaluating the best individuals 
    at the end of an experiment, run: run_best_candidates.py"""