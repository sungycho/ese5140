from abc import ABC
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import math
from utils import consensus as cn
import numpy as np
from scipy.spatial import distance_matrix
import copy
import heapq
rng = np.random

plt.rc('text', usetex=True)
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams["font.size"] = "12"


fig_fold='figures_part2/'
# Parameters
N=100 # number of agents
T= 10 # time (s)
# Part 1: Simulating centralized (optimal) controller

simulator = cn.ConsensusSimulator(N, T)
controller = cn.OptimalController()
ref_vel, est_vel, est_vels, biased_vels, accels = simulator.simulate(steps=100, controller=controller)


# Plots velocity for 100 agents
for i in range(0, 100):
  plt.plot(np.arange(0, int(T),0.1), np.sqrt(est_vels[0,:,0, i]** 2 + est_vels[0,:,0, i]**2))

#plt.title(r'Agent velocities $\|\bf{v}_{i,n}\|_2$ for ' + str(N)+ ' agents (centralized controller)')
plt.xlabel(r'$time (s)$')
plt.ylabel(r'$\|{\bf v}_{in}\|_2$')
plt.rcParams["figure.figsize"] = (10,6)
plt.grid()
plt.savefig(fig_fold+'2-2.png', dpi=300)
plt.show()

# Plots average velocity difference between simulations

results_centralized = np.zeros((10, 100))

# for sim in range(10):
#   for t in range(100):
#     curr_step = np.sqrt(est_vels[sim,t,0, :]** 2 + est_vels[sim,t,1, :]**2)

#     results_centralized[sim, t] = np.sqrt((curr_step - curr_step.mean())**2).mean()

# for sim in range(5):
#   plt.plot(results_centralized[sim, :60], label = 'Simulation ' + str(sim + 1))

# plt.legend()
# plt.title(r'$|\Delta \vec{v}|$ as a function of iterations for centralized controller')
# plt.grid()
# plt.xlabel('$Iteration $')
# plt.ylabel(r'$\Delta |\vec{v}|$')
# plt.savefig(fig_fold+'centralized.png', dpi=300)
# plt.show()

for sim in range(10):
  for t in range(100):
    curr_step = np.sqrt(est_vels[sim, t, 0, :] ** 2 + est_vels[sim, t, 1, :] ** 2)

    results_centralized[sim, t] = np.sqrt((curr_step - curr_step.mean()) ** 2).mean()

min_val, max_val = [], []
for t in range(100):
  min_val.append(min(results_centralized[:, t]))
  max_val.append(max(results_centralized[:, t]))

plt.plot(np.arange(0, T, 0.1),results_centralized.mean(axis=0), color='red',
                 label='Average velocity difference')

plt.fill_between(np.arange(0, T, 0.1), min_val, max_val, alpha=0.4, color='orange')

plt.legend()
plt.grid()
plt.xlabel('$time (s)$')
plt.ylabel(r'$\|\Delta  {\bf v}_{in}\|_2$')
#plt.title(r'Average, worst and best magnitude  of $\vec{v}$ for ' + str(N)+ ' agents (centralized controller)')
plt.savefig(fig_fold+'centralized.png', dpi=300)

plt.show()


#####################################################################
# Part 2: Simulating decentralized controller
# Plots different velocity plots for different K values.

D = 2
K = 4
nSamples = 200
N = 100

com_network = np.zeros((nSamples, N, N))

# Networks
for i in range(nSamples):
  com_network[i, :, :], _ = cn.agent_communication(100, 200, N, D)

simulator = cn.ConsensusSimulator(N, nSamples, duration=10)
controller = cn.DistributedController(com_network, K)
ref_vel, est_vel, est_vels, biased_vels, accels = simulator.simulate(steps=100, controller=controller)

for i in range(0, 100, 5):
  plt.plot(np.arange(0, 10,0.1), np.sqrt(est_vels[0, :, 0, i] ** 2 + est_vels[0, :, 0, i] ** 2))

plt.xlabel(r'$time (s)$')
plt.ylabel(r'$\|\bf{v}_{in}\|_2$')
plt.rcParams["figure.figsize"] = (10, 6)
#plt.title(r'Magnitude of $\|\bf{v}_{i,n}\|_2$ for ' + str(N)+ ' agents (decentralized controller) with degree ' + str (D)+' and K='+str(K))
plt.grid()
plt.savefig(fig_fold+'2-4-3.png', dpi=300)
plt.show()

# Plotting average velocity differences

results_decentralized = np.zeros((10, 100))

for sim in range(10):
  for t in range(100):
    curr_step = np.sqrt(est_vels[sim, t, 0, :] ** 2 + est_vels[sim, t, 1, :] ** 2)

    results_decentralized[sim, t] = np.sqrt((curr_step - curr_step.mean()) ** 2).mean()

min_val_dec, max_val_dec = [], []
for t in range(100):
  min_val_dec.append(min(results_decentralized[:, t]))
  max_val_dec.append(max(results_decentralized[:, t]))

plt.plot(np.arange(0, T, 0.1),results_decentralized.mean(axis=0), color='red',
                 label='Average velocity difference')

plt.fill_between(np.arange(0, T, 0.1), min_val_dec, max_val_dec, alpha=0.4, color='orange')

plt.legend()
plt.grid()
plt.xlabel('$time (s)$')
plt.ylabel(r'$\|\Delta  \bf{v}_{in}\|_2$')
#plt.title(r'Average, worst and best magnitude  of $\vec{v}$ for ' + str(N)+ ' agents (decentralized controller) with degree ' + str (D)+' and K='+str(K))
plt.savefig(fig_fold+'decentralized.png', dpi=300)

plt.show()
