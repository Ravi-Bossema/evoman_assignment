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


experiment_name = 'specialist_assignment_TS'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# --- Constants ---
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

N_HIDDEN_NEURONS = 10
ENEMY = 8
ENV = Environment(experiment_name=experiment_name,
                  enemies=[ENEMY],
                  playermode="ai",
                  player_controller=player_controller(N_HIDDEN_NEURONS),
                  enemymode="static",
                  level=2,
                  speed="fastest")

n_exp = 10

# Have the best solution play against a selected enemy
mean_individual_gain = []


for candidate in range(n_exp):
    individual_gain = []
    for i in range(5):
        sol = np.loadtxt(experiment_name + '/solutions_enemy' + str(ENEMY) + '/best_candidate_' + str(candidate)
                         + '.txt')
        result = ENV.play(sol)
        # Individual gain is defined by player energy - enemy energy
        individual_gain.append(result[1] - result[2])
    mean_individual_gain.append(statistics.mean(individual_gain))

# Plot the individual gain of the best candidate of the last generation for each experiment
fig, ax = plt.subplots(1)
bp = ax.boxplot(mean_individual_gain, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('blue')
ax.set_xticklabels(['EA using tournament selection'])
ax.set_ylim([60, 100])
ax.set_ylabel('Individual gain')
ax.set_title('Individual gain of best individuals over ' + str(n_exp) + ' experiments for enemy ' + str(ENEMY))
ax.grid()
plt.savefig(experiment_name + '/enemy' + str(ENEMY) + '_mean_individual_gain.png', dpi=300, bbox_inches='tight')
plt.show()