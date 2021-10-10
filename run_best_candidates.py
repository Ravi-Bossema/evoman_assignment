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

experiment_name_TS = 'specialist_assignment_TS'
experiment_name_RS = 'specialist_assignment_RS'

# --- Constants ---
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

N_HIDDEN_NEURONS = 10
ENEMY = 8
N_EXP = 10

ENV = Environment(experiment_name=experiment_name_TS,
                  enemies=[ENEMY],
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
            sol = np.loadtxt(experiment_name + '/solutions_enemy' + str(ENEMY) + '/best_candidate_' + str(candidate)
                             + '.txt')
            result = ENV.play(sol)
            # Individual gain is defined by player energy - enemy energy
            individual_gain.append(result[1] - result[2])
        mean_individual_gain.append(statistics.mean(individual_gain))
    return mean_individual_gain


mean_individual_gain_TS = average_individual_gain(experiment_name_TS)
mean_individual_gain_RS = average_individual_gain(experiment_name_RS)

# Plot the individual gain of the best candidate of the last generation for each experiment
fig, ax = plt.subplots(1)
bp = ax.boxplot([mean_individual_gain_TS, mean_individual_gain_RS], patch_artist=True)
colors = ['blue', 'red']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
ax.set_xticklabels(['Tournament selection', 'Random selection'])
ax.set_ylim([60, 100])
ax.set_ylabel('Individual gain')
ax.set_title('Individual gain of best individuals over ' + str(N_EXP) + ' experiments for enemy ' + str(ENEMY))
ax.grid()
plt.savefig(experiment_name_RS + '/enemy' + str(ENEMY) + '_mean_individual_gain.png', dpi=300, bbox_inches='tight')
print(stats.ttest_ind(mean_individual_gain_TS, mean_individual_gain_RS))
plt.show()