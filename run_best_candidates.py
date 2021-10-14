import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'evoman')
from demo_controller import player_controller

# imports other libs
import time
import statistics
import numpy as np
from scipy import stats
from math import fabs,sqrt
import glob, os
from environment import Environment

experiment_name_TS = 'generalist_assignment_TS'
experiment_name_PC = 'generalist_assignment_PC'

# --- Constants ---
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

N_HIDDEN_NEURONS = 10
TRAINED_ON = [6,7,8]
N_EXP = 10
RUN_TYPE = "Test_Best_Candidate"

enemy = [1,2,3,4,5,6,7,8]


ENV = Environment(experiment_name=experiment_name_TS,
                  enemies=enemy,
                  playermode="ai",
                  player_controller=player_controller(N_HIDDEN_NEURONS),
                  enemymode="static",
                  level=2,
                  speed="fastest")


def average_individual_gain(experiment_name):
    """Have the best candidate solutions play against the enemy 5 times and calculate the mean individual gain"""
    mean_individual_gain = []

    for candidate in range(N_EXP):
        individual_gain = []
        for i in range(5):
            sol = np.loadtxt(experiment_name + '/solutions_enemy' + str(TRAINED_ON) + '/best_candidate_' + str(candidate)
                             + '.txt')
            result = ENV.play(sol)
            # Individual gain is defined by player energy - enemy energy
            individual_gain.append(result[1] - result[2])
        mean_individual_gain.append(statistics.mean(individual_gain))
    return mean_individual_gain


def get_best_individual(experiment_name):
    """"Go through all the best candidates and select the one with
    the very best solution for playing against all enemies"""
    number_best_enemy = 0
    # Worst possible gain measure is -100
    gain_measure = -100
    for candidate in range(N_EXP):
        sol = np.loadtxt(experiment_name + '/solutions_enemy' + str(TRAINED_ON) + '/best_candidate_' + str(candidate)
                         + '.txt')
        result = ENV.play(sol)
        # Individual gain is defined by player energy - enemy energy
        individual_gain = result[1] - result[2]
        if individual_gain > gain_measure:
            gain_measure = individual_gain
            number_best_enemy = candidate
    return number_best_enemy, gain_measure


def test_best_individual(best_solution):
    """Returns a list of lists where each item is first the player life and then the enemy life"""
    sol = np.loadtxt(best_solution)
    performance = []
    for i in range(8):
        ENV.update_parameter('enemies', [i+1])
        player_life = []
        enemy_life = []
        performance_on_enemy = []
        for i in range(5):
            result = ENV.play(sol)
            player_life.append(result[1])
            enemy_life.append(result[2])
        performance_on_enemy.append(statistics.mean(player_life))
        performance_on_enemy.append(statistics.mean(enemy_life))
        performance.append(performance_on_enemy)
    return performance


# Code for Boxplot plotting
if RUN_TYPE == 'Boxplot':
    mean_individual_gain_TS = average_individual_gain(experiment_name_TS)
    mean_individual_gain_PC = average_individual_gain(experiment_name_PC)

    # Plot the individual gain of the best candidate of the last generation for each experiment
    fig, ax = plt.subplots(1)
    bp = ax.boxplot([mean_individual_gain_TS, mean_individual_gain_PC], patch_artist=True)
    colors = ['blue', 'red']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_xticklabels(['No Parameter Control', 'Parameter Control'])
    ax.set_ylim([-50, 30])
    ax.set_ylabel('Individual gain')
    ax.set_title('Individual gain of best individuals over ' + str(N_EXP) + ' experiments, trained on ' + str(TRAINED_ON))
    ax.grid()
    plt.savefig(experiment_name_PC + '/enemy' + str(TRAINED_ON) + '_mean_individual_gain.png', dpi=300, bbox_inches='tight')
    print(stats.ttest_ind(mean_individual_gain_TS, mean_individual_gain_PC))
    plt.show()

elif RUN_TYPE == 'Best_Candidate':
    best_individual_TS, best_gain_TS = get_best_individual(experiment_name_TS)
    best_individual_PC, best_gain_PC = get_best_individual(experiment_name_PC)
    # Comparing all values made it clear that the individual from the 10th experiment,
    # trained on [6,7,8] with parameter control was the best
    print("Best individual no PC " + str(best_individual_TS) + " with fitness " + str(best_gain_TS))
    print("Best individual with PC " + str(best_individual_PC) + " with fitness " + str(best_gain_PC))

else:
    best_performance = test_best_individual('generalist_assignment_PC/solutions_enemy[6, 7, 8]/best_candidate_9.txt')
    print(best_performance)
