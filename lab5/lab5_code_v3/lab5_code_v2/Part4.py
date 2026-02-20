from importlib import reload
import numpy as np
from utils import consensus
from utils import networks
import os
import torch;
import numpy as np
import matplotlib.pyplot as plt
torch.set_default_dtype(torch.float64)
import torch.nn as nn
import torch.optim as optim
from utils import gnnPart4 as gnn

# import training
# import model
import datetime
from copy import deepcopy

plt.rc('text', usetex=True)
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams["font.size"] = "12"


# %% Parameters
D = 2
K = 4
nSamples = 240
N = 100
T=10
step = int(T/0.1)

fig_fold = 'figures_part4/'
network = np.zeros((nSamples, N, N))
com_network = np.zeros((nSamples, step, N, N))

# %% Networks
for i in range(nSamples):
    network[i, :, :], com_network[i, 0, :, :] = networks.agent_communication(100, 200, N, D)
    for t in range(1, step):
        com_network[i, t, :, :] = deepcopy(com_network[i, 0, :, :])

# %% Dataset generation
optimal_controller = consensus.OptimalController()
decentralized_controller = consensus.DistributedController(network, K)
simulator = consensus.ConsensusSimulator(N, nSamples)
networkss, positions, ref_vel, est_vel, est_vels, biased_vels, accels = simulator.simulate_pos(100, 200, D, 100,
                                                                                               optimal_controller)

# %% Training
thisFilename = 'flockingGNN'
saveDirRoot = 'experiments'
saveDir = os.path.join(saveDirRoot, thisFilename)
today = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
saveDir = saveDir + '-%03d-' + today
# Create directory
if not os.path.exists(saveDir):
    os.makedirs(saveDir)

#   PyTorch seeds
torchState = torch.get_rng_state()
torchSeed = torch.initial_seed()
#   Numpy seeds
numpyState = np.random.RandomState().get_state()

nEpochs = 50  # Number of epochs
batchSize = 20  # Batch size
learningRate = 0.00025
beta1 = 0.9
beta2 = 0.999
hParamsLocalGNN = {}  # Hyperparameters (hParams) for the Local GNN
hParamsLocalGNN['name'] = 'LocalGNN'
hParamsLocalGNN['archit'] = gnn.LocalGNN_DB
hParamsLocalGNN['device'] = 'cuda:0' \
    if (True and torch.cuda.is_available()) \
    else 'cpu'
hParamsLocalGNN['dimNodeSignals'] = [4, 64]  # Features per layer
hParamsLocalGNN['nFilterTaps'] = [4]  # Number of filter taps
hParamsLocalGNN['bias'] = True
hParamsLocalGNN['nonlinearity'] = nn.Tanh
hParamsLocalGNN['dimReadout'] = [2]
hParamsLocalGNN['dimEdgeFeatures'] = 1

if torch.cuda.is_available():
    torch.cuda.empty_cache()
hParamsDict = deepcopy(hParamsLocalGNN)
thisName = hParamsDict.pop('name')
callArchit = hParamsDict.pop('archit')
thisDevice = hParamsDict.pop('device')

##############
# PARAMETERS #
##############

thisArchit = callArchit(**hParamsDict)
thisArchit.to(thisDevice)
thisOptim = optim.Adam(thisArchit.parameters(), lr=learningRate, betas=(beta1, beta2))
thisLossFunction = nn.MSELoss()
thisTrainer = gnn.TrainerFlocking
GraphNNs = gnn.Model(thisArchit, thisLossFunction, thisOptim, thisTrainer, thisDevice, thisName, saveDir)

# Train
GraphNNs.train(simulator, positions, ref_vel, est_vel, est_vels, biased_vels, accels, networkss, nEpochs, batchSize)

# %%Test
ref_vel_test = ref_vel[200:220]
est_vel_test = est_vel[200:220]
pos_test = positions[200:220]
_, _, est_vels_valid, biased_vels_valid, accels_valid = simulator.computeTrajectory_pos(GraphNNs.archit, pos_test,
                                                                                        ref_vel_test, est_vel_test, 100)
cost = simulator.cost(est_vels_valid, biased_vels_valid, accels_valid)
print(cost)
print(np.shape(est_vels))

#%% Plotting
for i in range(0, 20):
  plt.plot(np.arange(0, T, 0.1), np.sqrt(est_vels_valid[0, :, 0, i] ** 2 + est_vels_valid[0, :, 0, i] ** 2))

plt.xlabel(r'$time (s)$')
plt.ylabel(r'$\|{\bf v}_{in}\|_2$')
plt.rcParams["figure.figsize"] = (10, 6)
#plt.title(r'Magnitude of $\vec{v}$ for ' + str(N)+ ' agents (gnn controller) with degree ' + str (D)+' and K='+str(K))
plt.grid()
plt.savefig(fig_fold+'4_1.png', dpi=300)
plt.figure()

results_gnn = np.zeros((10, 100))

for sim in range(10):
  for t in range(100):
    curr_step = np.sqrt(est_vels_valid[sim, t, 0, :] ** 2 + est_vels_valid[sim, t, 1, :] ** 2)

    results_gnn[sim, t] = np.sqrt((curr_step - curr_step.mean()) ** 2).mean()

min_val_dec, max_val_dec = [], []
for t in range(100):
  min_val_dec.append(min(results_gnn[:, t]))
  max_val_dec.append(max(results_gnn[:, t]))

plt.plot(np.arange(0, T, 0.1),results_gnn.mean(axis=0), color='red',
                 label='Average velocity difference')

plt.fill_between(np.arange(0, T, 0.1), min_val_dec, max_val_dec, alpha=0.4, color='orange')

plt.legend()
plt.grid()
plt.xlabel('$time (s)$')
plt.ylabel(r'$\|\Delta  {\bf v}_{in}\|_2$')
#plt.title(r'Average, worst and best magnitude  of $\vec{v}$ for ' + str(N)+ ' agents (gnn controller) with degree ' + str (D)+' and K='+str(K))
plt.savefig(fig_fold+'gnn_4-1.png', dpi=300)

plt.show()