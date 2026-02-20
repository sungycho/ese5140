from importlib import reload
import numpy as np
import os
import torch;
from utils import networks
torch.set_default_dtype(torch.float64)
import torch.nn as nn
import torch.optim as optim
import datetime
from copy import deepcopy
from utils import consensus as cn
import matplotlib.pyplot as plt
import math
import copy
#####################################################################


#####################################################################
# Part 3: Simulating GNN controller

#####################################################################
# Part 3.1: No mobility

def LSIGF_DB(h, S, x, b=None):

    F = h.shape[0]
    E = h.shape[1]
    K = h.shape[2]
    G = h.shape[3]
    B = S.shape[0]
    T = S.shape[1]
    N = S.shape[3]

    x = x.reshape([B, T, 1, G, N]).repeat(1, 1, E, 1, 1)
    z = x.reshape([B, T, 1, E, G, N])
    for k in range(1 ,K):
        x, _ = torch.split(x, [ T -1, 1], dim = 1)
        zeroRow = torch.zeros(B, 1, E, G, N, dtype=x.dtype ,device=x.device)
        x = torch.cat((zeroRow, x), dim = 1)
        x = torch.matmul(x, S)
        xS = x.reshape(B, T, 1, E, G, N)
        z = torch.cat((z, xS), dim = 2)

    z = z.permute(0, 1, 5, 3, 2, 4)
    z = z.reshape(B, T, N, E* K * G)
    h = h.reshape(F, E * K * G)
    h = h.permute(1, 0)
    y = torch.matmul(z, h)
    y = y.permute(0, 1, 3, 2)
    if b is not None:
        y = y + b

    return y


class GraphFilter_DB(nn.Module):

    def __init__(self, G, F, K, E=1, bias=True):

        super().__init__()

        self.G = G
        self.F = F
        self.K = K
        self.E = E
        self.S = None
        self.weight = nn.parameter.Parameter(torch.Tensor(F, E, K, G))
        if bias:
            self.bias = nn.parameter.Parameter(torch.Tensor(F, 1))
        else:
            self.register_parameter('bias', None)
        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):

        stdv = 1. / math.sqrt(self.G * self.K)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def addGSO(self, S):

        assert len(S.shape) == 5
        assert S.shape[2] == self.E
        self.N = S.shape[3]
        assert S.shape[4] == self.N
        self.S = S

    def forward(self, x):

        assert len(x.shape) == 4
        B = x.shape[0]
        assert self.S.shape[0] == B
        T = x.shape[1]
        assert self.S.shape[1] == T
        assert x.shape[3] == self.N
        u = LSIGF_DB(self.weight, self.S, x, self.bias)
        return u


class LocalGNN_DB(nn.Module):

    def __init__(self,
                 dimNodeSignals, nFilterTaps, bias,
                 nonlinearity,
                 dimReadout,
                 dimEdgeFeatures):

        super().__init__()

        self.L = len(nFilterTaps)
        self.F = dimNodeSignals
        self.K = nFilterTaps
        self.E = dimEdgeFeatures
        self.bias = bias
        self.sigma = nonlinearity
        self.dimReadout = dimReadout

        gfl0 = []
        for l in range(self.L):
            gfl0.append(GraphFilter_DB(self.F[l], self.F[l + 1], self.K[l],
                                       self.E, self.bias))
            gfl0.append(self.sigma())
        self.GFL0 = nn.Sequential(*gfl0)

        fc0 = []
        if len(self.dimReadout) > 0:
            fc0.append(nn.Linear(self.F[-1], dimReadout[0], bias=self.bias))
            for l in range(len(dimReadout) - 1):
                fc0.append(self.sigma())
                fc0.append(nn.Linear(dimReadout[l], dimReadout[l + 1],
                                     bias=self.bias))

        self.Readout0 = nn.Sequential(*fc0)

    def splitForward(self, x, S):

        assert len(S.shape) == 4 or len(S.shape) == 5
        if len(S.shape) == 4:
            S = S.unsqueeze(2)
        for l in range(self.L):
            self.GFL0[2 * l].addGSO(S)
        yGFL0 = self.GFL0(x)
        y = yGFL0.permute(0, 1, 3, 2)
        y = self.Readout0(y)

        return y.permute(0, 1, 3, 2), yGFL0
        # B x T x dimReadout[-1] x N, B x T x dimFeatures[-1] x N

    def forward(self, x, S):

        output, _ = self.splitForward(x, S)

        return output


class Trainer:

    def __init__(self, model, simulator,positions, ref_vel, est_vel, est_vels, biased_vels, accels, S, nEpochs, batchSize):

        self.model = model
        self.positions = positions
        self.est_vel = est_vel
        self.ref_vel = ref_vel
        self.est_vels = est_vels
        self.biased_vels = biased_vels
        self.accels = accels
        self.S = S
        self.nTrain = 200
        self.nTest = 20
        self.nValid = 20
        self.nEpochs = nEpochs
        self.simulator = simulator
        self.validationInterval = self.nTrain // batchSize

        if self.nTrain < batchSize:
            self.nBatches = 1
            self.batchSize = [self.nTrain]
        elif self.nTrain % batchSize != 0:
            self.nBatches = np.ceil(self.nTrain / batchSize).astype(np.int64)
            self.batchSize = [batchSize] * self.nBatches
            while sum(batchSize) != self.nTrain:
                self.batchSize[-1] -= 1
        else:
            self.nBatches = np.int(self.nTrain / batchSize)
            self.batchSize = [batchSize] * self.nBatches

        self.batchIndex = np.cumsum(self.batchSize).tolist()
        self.batchIndex = [0] + self.batchIndex




class Model:

    def __init__(self,
                 architecture,
                 loss,
                 optimizer,
                 trainer,
                 device, name, saveDir):

        self.archit = architecture
        self.nParameters = 0
        for param in list(self.archit.parameters()):
            if len(param.shape) > 0:
                thisNParam = 1
                for p in range(len(param.shape)):
                    thisNParam *= param.shape[p]
                self.nParameters += thisNParam
            else:
                pass
        self.loss = loss
        self.optim = optimizer
        self.trainer = trainer
        self.device = device
        self.name = name
        self.saveDir = saveDir

    def train(self, simulator, positions, ref_vel, est_vel, est_vels, biased_vels, accels, com_network, nEpochs, batchSize):
        self.trainer = self.trainer(self, simulator, positions, ref_vel, est_vel, est_vels, biased_vels, accels, com_network, nEpochs, batchSize)
        return self.trainer.train()

    def save(self, label='', **kwargs):
        if 'saveDir' in kwargs.keys():
            saveDir = kwargs['saveDir']
        else:
            saveDir = self.saveDir
        saveModelDir = os.path.join(saveDir, 'savedModels')
        # Create directory savedModels if it doesn't exist yet:
        if not os.path.exists(saveModelDir):
            os.makedirs(saveModelDir)
        saveFile = os.path.join(saveModelDir, self.name)
        torch.save(self.archit.state_dict(), saveFile + 'Archit' + label + '.ckpt')
        torch.save(self.optim.state_dict(), saveFile + 'Optim' + label + '.ckpt')

    def load(self, label='', **kwargs):
        if 'loadFiles' in kwargs.keys():
            (architLoadFile, optimLoadFile) = kwargs['loadFiles']
        else:
            saveModelDir = os.path.join(self.saveDir, 'savedModels')
            architLoadFile = os.path.join(saveModelDir,
                                          self.name + 'Archit' + label + '.ckpt')
            optimLoadFile = os.path.join(saveModelDir,
                                         self.name + 'Optim' + label + '.ckpt')
        self.archit.load_state_dict(torch.load(architLoadFile))
        self.optim.load_state_dict(torch.load(optimLoadFile))


class TrainerFlocking(Trainer):

    def __init__(self, model, simulator, positions, ref_vel, est_vel, est_vels, biased_vels, accels, S, nEpochs,
                 batchSize):

        super().__init__(model, simulator, positions, ref_vel, est_vel, est_vels, biased_vels, accels, S, nEpochs,
                         batchSize)

    def train(self):

        nTrain = self.nTrain
        thisArchit = self.model.archit
        thisLoss = self.model.loss
        thisOptim = self.model.optim
        thisDevice = self.model.device
        epoch = 0
        pos_train = self.positions[0:self.nTrain]
        posDiff, posDistSq = networks.computeDifferences(pos_train)
        com_networks = copy.deepcopy(self.S[0:self.nTrain])
        com_networks = (np.abs(com_networks) > 1e-9).astype(self.positions.dtype)
        com_networks = np.expand_dims(com_networks, 2)
        posDiff = posDiff * com_networks
        statePos = np.sum(posDiff, axis=4)
        xTrainAll = np.concatenate(
            (self.est_vels[0:self.nTrain].copy(), self.biased_vels[0:self.nTrain].copy(), statePos), axis=2)
        yTrainAll = self.accels[0:self.nTrain].copy()
        StrainAll = self.S[0:self.nTrain].copy()

        while epoch < self.nEpochs:

            randomPermutation = np.random.permutation(nTrain)
            idxEpoch = [int(i) for i in randomPermutation]

            batch = 0
            while batch < self.nBatches:

                thisBatchIndices = idxEpoch[self.batchIndex[batch]: self.batchIndex[batch + 1]]
                xTrain = xTrainAll[thisBatchIndices]
                yTrain = yTrainAll[thisBatchIndices]
                Strain = StrainAll[thisBatchIndices]
                xTrain = torch.tensor(xTrain, device=thisDevice)
                Strain = torch.tensor(Strain, device=thisDevice)
                yTrain = torch.tensor(yTrain, device=thisDevice)
                startTime = datetime.datetime.now()
                thisArchit.zero_grad()
                yHatTrain = thisArchit(xTrain, Strain)
                lossValueTrain = thisLoss(yHatTrain, yTrain)
                lossValueTrain.backward()
                thisOptim.step()
                endTime = datetime.datetime.now()
                timeElapsed = abs(endTime - startTime).total_seconds()
                del xTrain
                del Strain
                del yTrain
                del lossValueTrain

                # \\\\\\\
                # \\\ VALIDATION
                # \\\\\\\

                if (epoch * self.nBatches + batch) % self.validationInterval == 0:

                    startTime = datetime.datetime.now()
                    ref_vel = self.ref_vel[220:240]
                    est_vel = self.est_vel[220:240]
                    position = self.positions[220:240]
                    _, _, est_vels_valid, biased_vels_valid, accels_valid, _ = self.simulator.computeTrajectory_pos_collision(
                        thisArchit, position, ref_vel, est_vel, 100)
                    accValid = self.simulator.cost(est_vels_valid, biased_vels_valid, accels_valid)
                    endTime = datetime.datetime.now()
                    timeElapsed = abs(endTime - startTime).total_seconds()

                    print("\t(E: %2d, B: %3d) %8.4f - %6.4fs" % (
                        epoch + 1, batch + 1,
                        accValid,
                        timeElapsed), end=' ')
                    print("[VALIDATION] \n", end='')

                    if epoch == 0 and batch == 0:
                        bestScore = accValid
                        bestEpoch, bestBatch = epoch, batch
                        self.model.save(label='Best')
                        # Start the counter
                    else:
                        thisValidScore = accValid
                        if thisValidScore < bestScore:
                            bestScore = thisValidScore
                            bestEpoch, bestBatch = epoch, batch
                            print("\t=> New best achieved: %.4f" % (bestScore))
                            self.model.save(label='Best')
                    del ref_vel
                    del est_vel
                batch += 1
            epoch += 1

        self.model.save(label='Last')
        #################
        # TRAINING OVER #
        #################
        if self.nEpochs == 0:
            self.model.save(label='Best')
            self.model.save(label='Last')
            print("\nWARNING: No training. Best and Last models are the same.\n")
        self.model.load(label='Best')
        if self.nEpochs > 0:
            print("\t=> Best validation achieved (E: %d, B: %d): %.4f" % (
                bestEpoch + 1, bestBatch + 1, bestScore))