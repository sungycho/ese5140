




import torch
import numpy as np
import datetime
import networks
import copy

class TrainerFlocking(Trainer):

    def __init__(self, model, simulator, positions, ref_vel, est_vel, est_vels, biased_vels, accels, S, nEpochs, batchSize):

        super().__init__(model, simulator, positions, ref_vel, est_vel, est_vels, biased_vels, accels, S, nEpochs, batchSize)

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
        statePos = np.sum(posDiff, axis = 4)
        xTrainAll = np.concatenate((self.est_vels[0:self.nTrain].copy(), self.biased_vels[0:self.nTrain].copy(), statePos), axis = 2)
        yTrainAll = self.accels[0:self.nTrain].copy()
        StrainAll = self.S[0:self.nTrain].copy()

        while epoch < self.nEpochs:

            randomPermutation = np.random.permutation(nTrain)
            idxEpoch = [int(i) for i in randomPermutation]

            batch = 0
            while batch < self.nBatches:

                thisBatchIndices = idxEpoch[self.batchIndex[batch] : self.batchIndex[batch+1]]
                xTrain = xTrainAll[thisBatchIndices]
                yTrain = yTrainAll[thisBatchIndices]
                Strain = StrainAll[thisBatchIndices]
                xTrain = torch.tensor(xTrain, device = thisDevice)
                Strain = torch.tensor(Strain, device = thisDevice)
                yTrain = torch.tensor(yTrain, device = thisDevice)
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

                #\\\\\\\
                #\\\ VALIDATION
                #\\\\\\\

                if (epoch * self.nBatches + batch) % self.validationInterval == 0:

                    startTime = datetime.datetime.now()
                    ref_vel = self.ref_vel[220:240]
                    est_vel = self.est_vel[220:240]
                    position = self.positions[220:240]
                    _, _, est_vels_valid, biased_vels_valid, accels_valid = self.simulator.computeTrajectory_pos_collision(thisArchit, position, ref_vel, est_vel, 100)
                    accValid = self.simulator.cost(est_vels_valid, biased_vels_valid, accels_valid)
                    endTime = datetime.datetime.now()
                    timeElapsed = abs(endTime - startTime).total_seconds()

                    print("\t(E: %2d, B: %3d) %8.4f - %6.4fs" % (
                            epoch+1, batch+1,
                            accValid,
                            timeElapsed), end = ' ')
                    print("[VALIDATION] \n", end = '')

                    if epoch == 0 and batch == 0:
                        bestScore = accValid
                        bestEpoch, bestBatch = epoch, batch
                        self.model.save(label = 'Best')
                        # Start the counter
                    else:
                        thisValidScore = accValid
                        if thisValidScore < bestScore:
                            bestScore = thisValidScore
                            bestEpoch, bestBatch = epoch, batch
                            print("\t=> New best achieved: %.4f" % (bestScore))
                            self.model.save(label = 'Best')
                    del ref_vel
                    del est_vel
                batch += 1
            epoch += 1

        self.model.save(label = 'Last')
        #################
        # TRAINING OVER #
        #################
        if self.nEpochs == 0:
            self.model.save(label = 'Best')
            self.model.save(label = 'Last')
            print("\nWARNING: No training. Best and Last models are the same.\n")
        self.model.load(label = 'Best')
        if self.nEpochs > 0:
            print("\t=> Best validation achieved (E: %d, B: %d): %.4f" % (
                    bestEpoch + 1, bestBatch + 1, bestScore))
#Here, for the validation, we similarly create the Python function  in the class  as:

    def computeTrajectory_pos_collision(self, archit, position, ref_vel, est_vel, step):
        
        batchSize = est_vel.shape[0]
        nAgents = est_vel.shape[2]
        ref_vel = ref_vel.squeeze(2)
        architDevice = list(archit.parameters())[0].device
        est_vels = np.zeros((batchSize, step, 2, nAgents), dtype = np.float)
        ref_vels = np.zeros((batchSize, step, 2), dtype = np.float)
        positions = np.zeros((batchSize, step, 2, nAgents), dtype = np.float)
        biased_ref_vels = np.zeros((batchSize, step, 2, nAgents), dtype = np.float)
        bias = np.zeros((batchSize, step, 2, nAgents))
        accels = np.zeros((batchSize, step, 2, nAgents), dtype=np.float)
        states = np.zeros((batchSize, step, 6, nAgents), dtype=np.float)
        graphs = np.zeros((batchSize, step, nAgents, nAgents), dtype = np.float)
        com_networks = np.zeros((batchSize, step, nAgents, nAgents), dtype = np.float)           
        est_vels[:,0,:,:] = est_vel.copy()
        ref_vels[:,0,:] = ref_vel.copy()
        bias[:,0,:,:] = np.random.normal(loc=0,scale=(4 * np.sqrt(2/np.pi)),size=(batchSize, 2, nAgents))
        ref_vel_temp = np.expand_dims(ref_vels[:,0,:], 2)
        biased_ref_vels[:,0,:,:] = ref_vel_temp + bias[:,0,:,:]
        positions[:,0,:,:] = position[:,0,:,:]      

        for i in range(20):
            com_networks[i,0,:,:], graphs[i,0,:,:] = networks.agent_communication_pos(positions[i,0,:,:], nAgents, 2)
        
        for t in range(1, step):
            
            posDiff, posDistSq = networks.computeDifferences(positions[:,t-1,:,:])
            com_network = copy.deepcopy(com_networks[:,t-1,:,:])
            com_network = np.expand_dims(com_network, 1)
            posDiff = posDiff * com_network
            statePos = np.sum(posDiff, axis = 3)
            statePos = np.expand_dims(statePos, 1)               
            thisbiased_ref_vels = np.expand_dims(biased_ref_vels[:,t-1,:,:], 1)
            thisest_vels = np.expand_dims(est_vels[:,t-1,:,:], 1)
            thisState = np.concatenate((thisest_vels, thisbiased_ref_vels, statePos), axis = 2)
            states[:,t-1,:,:] = thisState.squeeze(1)

            x = torch.tensor(states[:,0:t,:,:], device = architDevice)
            S = torch.tensor(graphs[:,0:t,:,:], device = architDevice)
            with torch.no_grad():
                thisaccels = archit(x, S)
            thisaccels = thisaccels.cpu().numpy()[:,-1,:,:]
            thisaccels[thisaccels > 3] = 3
            thisaccels[thisaccels < -3] = -3
            accels[:,t-1,:,:] = thisaccels
            est_vels[:,t,:,:] = accels[:,t-1,:,:] * 0.1 +est_vels[:,t-1,:,:]
            positions[:,t,:,:] = accels[:,t-1,:,:] * (0.1 ** 2)/2 + est_vels[:,t-1,:,:] * 0.1 + positions[:,t-1,:,:]
            ref_accel = np.random.normal(loc=0,scale=(self.ref_vel_delta * np.sqrt(2/np.pi)),size=(batchSize, 2))
            ref_vels[:,t,:] = ref_vels[:,t-1,:] + ref_accel * 0.1
            ref_vel_temp = np.expand_dims(ref_vels[:,t,:], 2)
            biased_ref_vels[:,t,:,:] = ref_vel_temp + bias[:,0,:,:]
            for i in range(20):
                com_networks[i,t,:,:], graphs[i,t,:,:] = networks.agent_communication_pos(positions[i,t,:,:], nAgents, 2)   

        posDiff, posDistSq = networks.computeDifferences(positions[:,-1,:,:])
        com_network = copy.deepcopy(com_networks[:,-1,:,:])
        com_network = np.expand_dims(com_network, 1)
        posDiff = posDiff * com_network
        statePos = np.sum(posDiff, axis = 3)
        statePos = np.expand_dims(statePos, 1)                  
        thisbiased_ref_vels = np.expand_dims(biased_ref_vels[:,-1,:,:], 1)
        thisest_vels = np.expand_dims(est_vels[:,-1,:,:], 1)
        thisState = np.concatenate((thisest_vels, thisbiased_ref_vels, statePos), axis = 2)
        states[:,-1,:,:] = thisState.squeeze(1)
        x = torch.tensor(states, device = architDevice)
        S = torch.tensor(graphs, device = architDevice)
        with torch.no_grad():
            thisaccels = archit(x, S)
        thisaccels = thisaccels.cpu().numpy()[:,-1,:,:]
        thisaccels[thisaccels > 3] = 3
        thisaccels[thisaccels < -3] = -3
        accels[:,-1,:,:] = thisaccels
            
        return ref_vel, est_vel, est_vels, biased_ref_vels, accels

#####################################################################
# Training loop  
