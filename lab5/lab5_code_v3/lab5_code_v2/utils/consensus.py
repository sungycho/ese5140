from abc import ABC
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import math
from scipy.spatial import distance_matrix
import copy
import heapq
rng = np.random
from utils import networks

class ConsensusController(ABC):
    def __call__(self, est_vel, biased_vel):
        pass


class OptimalController(ConsensusController):
    def __init__(self, sample_time=0.1, energy_weight=1):
        self.sample_time = sample_time
        self.energy_weight = energy_weight

    def __call__(self, est_vel, biased_vel, step):
        mean_biased_vel = np.mean(biased_vel[:, step, :, :], axis=2)
        mean_biased_vel = np.expand_dims(mean_biased_vel, 2)

        return -(est_vel[:, step, :, :] - mean_biased_vel) / (self.sample_time * (1 + self.energy_weight))


class ConsensusSimulator:
    def __init__(
            self,
            N,
            nSamples,
            initial_ref_vel=1.0,
            ref_vel_delta=1.0,
            ref_vel_bias=4.0,
            ref_est_delta=1.0,
            max_agent_acceleration=3.0,
            sample_time=0.1,
            duration=10,
            energy_weight=1,
            is_bias_constant=True
    ):

        # Simulator elements (see table on Lab instructions)
        self.N = N
        self.nSamples = nSamples
        self.initial_ref_vel = initial_ref_vel
        self.ref_vel_delta = ref_vel_delta
        self.ref_vel_bias = ref_vel_bias
        self.ref_est_delta = ref_est_delta
        self.max_agent_acceleration = max_agent_acceleration
        self.sample_time = sample_time
        self.tSamples = int(duration / self.sample_time)
        self.energy_weight = energy_weight
        self.is_bias_constant = is_bias_constant

    def random_vector(self, expected_norm, N, samples):
        # Wrapper for creating a random vector (following a Gaussian)
        return np.random.normal(0, (2 / 3.14) ** 0.5 * expected_norm, (samples, 2, N))

    def bias(self):
        return self.random_vector(self.ref_vel_bias, self.N, self.nSamples)

    def initial(self):
        ref_vel = self.random_vector(self.initial_ref_vel, 1, self.nSamples)
        est_vel = ref_vel + self.random_vector(self.ref_est_delta, self.N, self.nSamples)
        return ref_vel, est_vel

    def simulate(self, steps, controller: ConsensusController):

        ref_vel, est_vel = self.initial()
        bias = self.bias()

        # Arrays to store the elements of the simulation
        ref_vels = np.zeros((self.nSamples, self.tSamples, 2, 1))
        est_vels = np.zeros((self.nSamples, self.tSamples, 2, self.N))
        biased_vels = np.zeros((self.nSamples, self.tSamples, 2, self.N))
        accels = np.zeros((self.nSamples, self.tSamples, 2, self.N))

        # Initial conditions
        ref_vels[:, 0, :, :] = ref_vel
        est_vels[:, 0, :, :] = est_vel
        biased_vels[:, 0, :, :] = bias + ref_vel
        accels[:, 0, :, :] = controller(est_vels, biased_vels, 0)

        # Guarantees that acceleration is clipped to the max acceleration allowed
        this_accel = accels[:, 0, :, :].copy()
        this_accel[accels[:, 0, :, :] > self.max_agent_acceleration] = self.max_agent_acceleration
        this_accel[accels[:, 0, :, :] < -self.max_agent_acceleration] = -self.max_agent_acceleration
        accels[:, 0, :, :] = this_accel

        # Each time step
        for step in range(1, steps):
            ref_accel = self.random_vector(self.ref_vel_delta, 1, self.nSamples)
            ref_vels[:, step, :, :] = ref_vels[:, step - 1, :, :] + ref_accel * self.sample_time
            est_vels[:, step, :, :] = accels[:, step - 1, :, :] * self.sample_time + est_vels[:, step - 1, :, :]
            biased_vels[:, step, :, :] = bias + ref_vels[:, step, :, :]

            # Controller takes in the state and outputs the new acceleration
            accels[:, step, :, :] = controller(est_vels, biased_vels, step)

            this_accel = accels[:, step, :, :].copy()
            this_accel[accels[:, step, :, :] > self.max_agent_acceleration] = self.max_agent_acceleration
            this_accel[accels[:, step, :, :] < -self.max_agent_acceleration] = -self.max_agent_acceleration
            accels[:, step, :, :] = this_accel

        return ref_vel, est_vel, est_vels, biased_vels, accels

    def simulate_pos(self, wx, wy, degree, steps, controller: ConsensusController):

        ref_vel, est_vel = self.initial()
        bias = self.bias()
        x_pos = np.random.uniform(0, wx, (self.nSamples, 1, self.N))
        y_pos = np.random.uniform(0, wy, (self.nSamples, 1, self.N))
        pos = np.concatenate((x_pos, y_pos), axis=1)
        ref_vels = np.zeros((self.nSamples, self.tSamples, 2, 1))
        est_vels = np.zeros((self.nSamples, self.tSamples, 2, self.N))
        biased_vels = np.zeros((self.nSamples, self.tSamples, 2, self.N))
        accels = np.zeros((self.nSamples, self.tSamples, 2, self.N))
        positions = np.zeros((self.nSamples, self.tSamples, 2, self.N))
        networkss = np.zeros((self.nSamples, self.tSamples, self.N, self.N))
        ref_vels[:, 0, :, :] = ref_vel
        est_vels[:, 0, :, :] = est_vel
        biased_vels[:, 0, :, :] = bias + ref_vel
        accels[:, 0, :, :] = controller(est_vels, biased_vels, 0)
        positions[:, 0, :, :] = pos
        for i in range(self.nSamples):
            _, networkss[i, 0, :, :] = agent_communication_pos(positions[i, 0, :, :], self.N, degree)
        this_accel = accels[:, 0, :, :].copy()
        this_accel[accels[:, 0, :, :] > self.max_agent_acceleration] = self.max_agent_acceleration
        this_accel[accels[:, 0, :, :] < -self.max_agent_acceleration] = -self.max_agent_acceleration
        accels[:, 0, :, :] = this_accel

        for step in range(1, steps):
            ref_accel = self.random_vector(self.ref_vel_delta, 1, self.nSamples)
            ref_vels[:, step, :, :] = ref_vels[:, step - 1, :, :] + ref_accel * self.sample_time
            est_vels[:, step, :, :] = accels[:, step - 1, :, :] * self.sample_time + est_vels[:, step - 1, :, :]
            positions[:, step, :, :] = accels[:, step - 1, :, :] * (self.sample_time ** 2) / 2 + est_vels[:, step - 1,
                                                                                                 :,
                                                                                                 :] * self.sample_time + positions[
                                                                                                                         :,
                                                                                                                         step - 1,
                                                                                                                         :,
                                                                                                                         :]
            for i in range(self.nSamples):
                _, networkss[i, step, :, :] = agent_communication_pos(positions[i, step, :, :], self.N, degree)
            biased_vels[:, step, :, :] = bias + ref_vels[:, step, :, :]
            accels[:, step, :, :] = controller(est_vels, biased_vels, step)
            this_accel = accels[:, step, :, :].copy()
            this_accel[accels[:, step, :, :] > self.max_agent_acceleration] = self.max_agent_acceleration
            this_accel[accels[:, step, :, :] < -self.max_agent_acceleration] = -self.max_agent_acceleration
            accels[:, step, :, :] = this_accel

        return networkss, positions, ref_vel, est_vel, est_vels, biased_vels, accels

    def simulate_pos_collision(self, wx, wy, degree, steps, controller: ConsensusController):

        ref_vel, est_vel = self.initial()
        bias = self.bias()
        x_pos = np.random.uniform(0, wx, (self.nSamples, 1, self.N))
        y_pos = np.random.uniform(0, wy, (self.nSamples, 1, self.N))
        pos = np.concatenate((x_pos, y_pos), axis=1)
        ref_vels = np.zeros((self.nSamples, self.tSamples, 2, 1))
        est_vels = np.zeros((self.nSamples, self.tSamples, 2, self.N))
        biased_vels = np.zeros((self.nSamples, self.tSamples, 2, self.N))
        accels = np.zeros((self.nSamples, self.tSamples, 2, self.N))
        positions = np.zeros((self.nSamples, self.tSamples, 2, self.N))
        networkss = np.zeros((self.nSamples, self.tSamples, self.N, self.N))
        ref_vels[:, 0, :, :] = ref_vel
        est_vels[:, 0, :, :] = est_vel
        biased_vels[:, 0, :, :] = bias + ref_vel
        positions[:, 0, :, :] = pos
        accels[:, 0, :, :] = controller(10, 2, positions, est_vels, biased_vels, 0)

        for i in range(self.nSamples):
            _, networkss[i, 0, :, :] = networks.agent_communication_pos(positions[i, 0, :, :], self.N, degree)
        this_accel = accels[:, 0, :, :].copy()
        this_accel[accels[:, 0, :, :] > self.max_agent_acceleration] = self.max_agent_acceleration
        this_accel[accels[:, 0, :, :] < -self.max_agent_acceleration] = -self.max_agent_acceleration
        accels[:, 0, :, :] = this_accel

        for step in range(1, steps):
            ref_accel = self.random_vector(self.ref_vel_delta, 1, self.nSamples)
            ref_vels[:, step, :, :] = ref_vels[:, step - 1, :, :] + ref_accel * self.sample_time
            est_vels[:, step, :, :] = accels[:, step - 1, :, :] * self.sample_time + est_vels[:, step - 1, :, :]
            positions[:, step, :, :] = accels[:, step - 1, :, :] * (self.sample_time ** 2) / 2 + est_vels[:, step - 1,
                                                                                                 :,
                                                                                                 :] * self.sample_time + positions[
                                                                                                                         :,
                                                                                                                         step - 1,
                                                                                                                         :,
                                                                                                                         :]
            for i in range(self.nSamples):
                _, networkss[i, step, :, :] = networks.agent_communication_pos(positions[i, step, :, :], self.N, degree)
            biased_vels[:, step, :, :] = bias + ref_vels[:, step, :, :]
            accels[:, step, :, :] = controller(10, 2, positions, est_vels, biased_vels, step)
            this_accel = accels[:, step, :, :].copy()
            this_accel[accels[:, step, :, :] > self.max_agent_acceleration] = self.max_agent_acceleration
            this_accel[accels[:, step, :, :] < -self.max_agent_acceleration] = -self.max_agent_acceleration
            accels[:, step, :, :] = this_accel

        return networkss, positions, ref_vel, est_vel, est_vels, biased_vels, accels

    def cost(self, est_vels, biased_vels, accels):
        biased_vel_mean = np.mean(biased_vels, axis=3)
        biased_vel_mean = np.expand_dims(biased_vel_mean, 3)
        vel_error = est_vels - biased_vel_mean
        vel_error_norm = np.square(np.linalg.norm(vel_error, ord=2, axis=2, keepdims=False))
        vel_error_norm_mean = np.sum(np.mean(vel_error_norm, axis=2), axis=1)

        accel_norm = np.square(np.linalg.norm(self.sample_time * accels, ord=2, axis=2, keepdims=False))
        accel_norm_mean = np.sum(np.mean(accel_norm, axis=2), axis=1)

        cost = np.mean(vel_error_norm_mean / 2 + accel_norm_mean / 2)

        return cost

    def computeTrajectory_pos(self, archit, position, ref_vel, est_vel, step):

        batchSize = est_vel.shape[0]
        nAgents = est_vel.shape[2]
        ref_vel = ref_vel.squeeze(2)
        architDevice = list(archit.parameters())[0].device
        est_vels = np.zeros((batchSize, step, 2, nAgents), dtype=np.float)
        ref_vels = np.zeros((batchSize, step, 2), dtype=np.float)
        positions = np.zeros((batchSize, step, 2, nAgents), dtype=np.float)
        biased_ref_vels = np.zeros((batchSize, step, 2, nAgents), dtype=np.float)
        bias = np.zeros((batchSize, step, 2, nAgents))
        accels = np.zeros((batchSize, step, 2, nAgents), dtype=np.float)
        states = np.zeros((batchSize, step, 4, nAgents), dtype=np.float)
        graphs = np.zeros((batchSize, step, nAgents, nAgents), dtype=np.float)
        est_vels[:, 0, :, :] = est_vel.copy()
        ref_vels[:, 0, :] = ref_vel.copy()
        bias[:, 0, :, :] = np.random.normal(loc=0, scale=(4 * np.sqrt(2 / np.pi)), size=(batchSize, 2, nAgents))
        ref_vel_temp = np.expand_dims(ref_vels[:, 0, :], 2)
        biased_ref_vels[:, 0, :, :] = ref_vel_temp + bias[:, 0, :, :]
        positions[:, 0, :, :] = position[:, 0, :, :]

        for i in range(20):
            _, graphs[i, 0, :, :] = agent_communication_pos(positions[i, 0, :, :], nAgents, 2)

        for t in range(1, step):
            thisbiased_ref_vels = np.expand_dims(biased_ref_vels[:, t - 1, :, :], 1)
            thisest_vels = np.expand_dims(est_vels[:, t - 1, :, :], 1)
            thisState = np.concatenate((thisest_vels, thisbiased_ref_vels), axis=2)
            states[:, t - 1, :, :] = thisState.squeeze(1)

            x = torch.tensor(states[:, 0:t, :, :], device=architDevice)
            S = torch.tensor(graphs[:, 0:t, :, :], device=architDevice)
            with torch.no_grad():
                thisaccels = archit(x, S)
            thisaccels = thisaccels.cpu().numpy()[:, -1, :, :]
            thisaccels[thisaccels > 3] = 3
            thisaccels[thisaccels < -3] = -3
            accels[:, t - 1, :, :] = thisaccels
            est_vels[:, t, :, :] = accels[:, t - 1, :, :] * 0.1 + est_vels[:, t - 1, :, :]
            positions[:, t, :, :] = accels[:, t - 1, :, :] * (0.1 ** 2) / 2 + est_vels[:, t - 1, :,
                                                                              :] * 0.1 + positions[:, t - 1, :, :]
            ref_accel = np.random.normal(loc=0, scale=(self.ref_vel_delta * np.sqrt(2 / np.pi)), size=(batchSize, 2))
            ref_vels[:, t, :] = ref_vels[:, t - 1, :] + ref_accel * 0.1
            ref_vel_temp = np.expand_dims(ref_vels[:, t, :], 2)
            biased_ref_vels[:, t, :, :] = ref_vel_temp + bias[:, 0, :, :]
            for i in range(20):
                _, graphs[i, t, :, :] = agent_communication_pos(positions[i, t, :, :], nAgents, 2)

        thisbiased_ref_vels = np.expand_dims(biased_ref_vels[:, -1, :, :], 1)
        thisest_vels = np.expand_dims(est_vels[:, -1, :, :], 1)
        thisState = np.concatenate((thisest_vels, thisbiased_ref_vels), axis=2)
        states[:, -1, :, :] = thisState.squeeze(1)
        x = torch.tensor(states, device=architDevice)
        S = torch.tensor(graphs, device=architDevice)
        with torch.no_grad():
            thisaccels = archit(x, S)
        thisaccels = thisaccels.cpu().numpy()[:, -1, :, :]
        thisaccels[thisaccels > 3] = 3
        thisaccels[thisaccels < -3] = -3
        accels[:, -1, :, :] = thisaccels

        return ref_vel, est_vel, est_vels, biased_ref_vels, accels

    def computeTrajectory(self, archit, ref_vel, est_vel, step):

        batchSize = est_vel.shape[0]
        nAgents = est_vel.shape[2]
        ref_vel = ref_vel.squeeze(2)
        architDevice = list(archit.parameters())[0].device
        est_vels = np.zeros((batchSize, step, 2, nAgents), dtype=np.float)
        ref_vels = np.zeros((batchSize, step, 2), dtype=np.float)
        biased_ref_vels = np.zeros((batchSize, step, 2, nAgents), dtype=np.float)
        bias = np.zeros((batchSize, step, 2, nAgents))
        accels = np.zeros((batchSize, step, 2, nAgents), dtype=np.float)
        states = np.zeros((batchSize, step, 4, nAgents), dtype=np.float)
        graphs = np.zeros((batchSize, step, nAgents, nAgents), dtype=np.float)

        est_vels[:, 0, :, :] = est_vel.copy()
        ref_vels[:, 0, :] = ref_vel.copy()
        bias[:, 0, :, :] = np.random.normal(loc=0, scale=(4 * np.sqrt(2 / np.pi)), size=(batchSize, 2, nAgents))
        ref_vel_temp = np.expand_dims(ref_vels[:, 0, :], 2)
        biased_ref_vels[:, 0, :, :] = ref_vel_temp + bias[:, 0, :, :]

        for i in range(20):
            _, graphs[i, 0, :, :] = agent_communication(100, 200, self.N, 2)

        for t in range(1, step):
            thisbiased_ref_vels = np.expand_dims(biased_ref_vels[:, t - 1, :, :], 1)
            thisest_vels = np.expand_dims(est_vels[:, t - 1, :, :], 1)
            thisState = np.concatenate((thisest_vels, thisbiased_ref_vels), axis=2)
            states[:, t - 1, :, :] = thisState.squeeze(1)

            x = torch.tensor(states[:, 0:t, :, :], device=architDevice)
            S = torch.tensor(graphs[:, 0:t, :, :], device=architDevice)
            with torch.no_grad():
                thisaccels = archit(x, S)
            thisaccels = thisaccels.cpu().numpy()[:, -1, :, :]
            thisaccels[thisaccels > 3] = 3
            thisaccels[thisaccels < -3] = -3
            accels[:, t - 1, :, :] = thisaccels
            est_vels[:, t, :, :] = accels[:, t - 1, :, :] * 0.1 + est_vels[:, t - 1, :, :]
            ref_accel = np.random.normal(loc=0, scale=(self.ref_vel_delta * np.sqrt(2 / np.pi)), size=(batchSize, 2))
            ref_vels[:, t, :] = ref_vels[:, t - 1, :] + ref_accel * 0.1
            ref_vel_temp = np.expand_dims(ref_vels[:, t, :], 2)
            biased_ref_vels[:, t, :, :] = ref_vel_temp + bias[:, 0, :, :]
            graphs[:, t, :, :] = copy.deepcopy(graphs[:, 0, :, :])

        thisbiased_ref_vels = np.expand_dims(biased_ref_vels[:, -1, :, :], 1)
        thisest_vels = np.expand_dims(est_vels[:, -1, :, :], 1)
        thisState = np.concatenate((thisest_vels, thisbiased_ref_vels), axis=2)
        states[:, -1, :, :] = thisState.squeeze(1)
        x = torch.tensor(states, device=architDevice)
        S = torch.tensor(graphs, device=architDevice)
        with torch.no_grad():
            thisaccels = archit(x, S)
        thisaccels = thisaccels.cpu().numpy()[:, -1, :, :]
        thisaccels[thisaccels > 3] = 3
        thisaccels[thisaccels < -3] = -3
        accels[:, -1, :, :] = thisaccels

        return ref_vel, est_vel, est_vels, biased_ref_vels, accels

    def simulate_pos_collision(self, wx, wy, degree, steps, controller: ConsensusController):

        ref_vel, est_vel = self.initial()
        bias = self.bias()
        x_pos = np.random.uniform(0, wx, (self.nSamples, 1, self.N))
        y_pos = np.random.uniform(0, wy, (self.nSamples, 1, self.N))
        pos = np.concatenate((x_pos, y_pos), axis=1)
        ref_vels = np.zeros((self.nSamples, self.tSamples, 2, 1))
        est_vels = np.zeros((self.nSamples, self.tSamples, 2, self.N))
        biased_vels = np.zeros((self.nSamples, self.tSamples, 2, self.N))
        accels = np.zeros((self.nSamples, self.tSamples, 2, self.N))
        positions = np.zeros((self.nSamples, self.tSamples, 2, self.N))
        networkss = np.zeros((self.nSamples, self.tSamples, self.N, self.N))
        ref_vels[:, 0, :, :] = ref_vel
        est_vels[:, 0, :, :] = est_vel
        biased_vels[:, 0, :, :] = bias + ref_vel
        positions[:, 0, :, :] = pos
        accels[:, 0, :, :] = controller(10, 2, positions, est_vels, biased_vels, 0)

        for i in range(self.nSamples):
            _, networkss[i, 0, :, :] = agent_communication_pos(positions[i, 0, :, :], self.N, degree)
        this_accel = accels[:, 0, :, :].copy()
        this_accel[accels[:, 0, :, :] > self.max_agent_acceleration] = self.max_agent_acceleration
        this_accel[accels[:, 0, :, :] < -self.max_agent_acceleration] = -self.max_agent_acceleration
        accels[:, 0, :, :] = this_accel

        for step in range(1, steps):
            ref_accel = self.random_vector(self.ref_vel_delta, 1, self.nSamples)
            ref_vels[:, step, :, :] = ref_vels[:, step - 1, :, :] + ref_accel * self.sample_time
            est_vels[:, step, :, :] = accels[:, step - 1, :, :] * self.sample_time + est_vels[:, step - 1, :, :]
            positions[:, step, :, :] = accels[:, step - 1, :, :] * (self.sample_time ** 2) / 2 + est_vels[:, step - 1,
                                                                                                 :,
                                                                                                 :] * self.sample_time + positions[
                                                                                                                         :,
                                                                                                                         step - 1,
                                                                                                                         :,
                                                                                                                         :]
            for i in range(self.nSamples):
                _, networkss[i, step, :, :] = agent_communication_pos(positions[i, step, :, :], self.N, degree)
            biased_vels[:, step, :, :] = bias + ref_vels[:, step, :, :]
            accels[:, step, :, :] = controller(10, 2, positions, est_vels, biased_vels, step)
            this_accel = accels[:, step, :, :].copy()
            this_accel[accels[:, step, :, :] > self.max_agent_acceleration] = self.max_agent_acceleration
            this_accel[accels[:, step, :, :] < -self.max_agent_acceleration] = -self.max_agent_acceleration
            accels[:, step, :, :] = this_accel

        return networkss, positions, ref_vel, est_vel, est_vels, biased_vels, accels

    #####################################################################
    def computeTrajectory_pos_collision(self, archit, position, ref_vel, est_vel, step):

        batchSize = est_vel.shape[0]
        nAgents = est_vel.shape[2]
        ref_vel = ref_vel.squeeze(2)
        architDevice = list(archit.parameters())[0].device
        est_vels = np.zeros((batchSize, step, 2, nAgents), dtype=np.float)
        ref_vels = np.zeros((batchSize, step, 2), dtype=np.float)
        positions = np.zeros((batchSize, step, 2, nAgents), dtype=np.float)
        biased_ref_vels = np.zeros((batchSize, step, 2, nAgents), dtype=np.float)
        bias = np.zeros((batchSize, step, 2, nAgents))
        accels = np.zeros((batchSize, step, 2, nAgents), dtype=np.float)
        states = np.zeros((batchSize, step, 6, nAgents), dtype=np.float)
        graphs = np.zeros((batchSize, step, nAgents, nAgents), dtype=np.float)
        com_networks = np.zeros((batchSize, step, nAgents, nAgents), dtype=np.float)
        est_vels[:, 0, :, :] = est_vel.copy()
        ref_vels[:, 0, :] = ref_vel.copy()
        bias[:, 0, :, :] = np.random.normal(loc=0, scale=(4 * np.sqrt(2 / np.pi)), size=(batchSize, 2, nAgents))
        ref_vel_temp = np.expand_dims(ref_vels[:, 0, :], 2)
        biased_ref_vels[:, 0, :, :] = ref_vel_temp + bias[:, 0, :, :]
        positions[:, 0, :, :] = position[:, 0, :, :]

        for i in range(20):
            com_networks[i, 0, :, :], graphs[i, 0, :, :] = networks.agent_communication_pos(positions[i, 0, :, :],
                                                                                            nAgents, 2)

        for t in range(1, step):

            posDiff, posDistSq = networks.computeDifferences(positions[:, t - 1, :, :])
            com_network = copy.deepcopy(com_networks[:, t - 1, :, :])
            com_network = np.expand_dims(com_network, 1)
            posDiff = posDiff * com_network
            statePos = np.sum(posDiff, axis=3)
            statePos = np.expand_dims(statePos, 1)
            thisbiased_ref_vels = np.expand_dims(biased_ref_vels[:, t - 1, :, :], 1)
            thisest_vels = np.expand_dims(est_vels[:, t - 1, :, :], 1)
            thisState = np.concatenate((thisest_vels, thisbiased_ref_vels, statePos), axis=2)
            states[:, t - 1, :, :] = thisState.squeeze(1)

            x = torch.tensor(states[:, 0:t, :, :], device=architDevice)
            S = torch.tensor(graphs[:, 0:t, :, :], device=architDevice)
            with torch.no_grad():
                thisaccels = archit(x, S)
            thisaccels = thisaccels.cpu().numpy()[:, -1, :, :]
            thisaccels[thisaccels > 3] = 3
            thisaccels[thisaccels < -3] = -3
            accels[:, t - 1, :, :] = thisaccels
            est_vels[:, t, :, :] = accels[:, t - 1, :, :] * 0.1 + est_vels[:, t - 1, :, :]
            positions[:, t, :, :] = accels[:, t - 1, :, :] * (0.1 ** 2) / 2 + est_vels[:, t - 1, :,
                                                                              :] * 0.1 + positions[:, t - 1, :, :]
            ref_accel = np.random.normal(loc=0, scale=(self.ref_vel_delta * np.sqrt(2 / np.pi)), size=(batchSize, 2))
            ref_vels[:, t, :] = ref_vels[:, t - 1, :] + ref_accel * 0.1
            ref_vel_temp = np.expand_dims(ref_vels[:, t, :], 2)
            biased_ref_vels[:, t, :, :] = ref_vel_temp + bias[:, 0, :, :]
            for i in range(20):
                com_networks[i, t, :, :], graphs[i, t, :, :] = networks.agent_communication_pos(positions[i, t, :, :],
                                                                                                nAgents, 2)

        posDiff, posDistSq = networks.computeDifferences(positions[:, -1, :, :])
        com_network = copy.deepcopy(com_networks[:, -1, :, :])
        com_network = np.expand_dims(com_network, 1)
        posDiff = posDiff * com_network
        statePos = np.sum(posDiff, axis=3)
        statePos = np.expand_dims(statePos, 1)
        thisbiased_ref_vels = np.expand_dims(biased_ref_vels[:, -1, :, :], 1)
        thisest_vels = np.expand_dims(est_vels[:, -1, :, :], 1)
        thisState = np.concatenate((thisest_vels, thisbiased_ref_vels, statePos), axis=2)
        states[:, -1, :, :] = thisState.squeeze(1)
        x = torch.tensor(states, device=architDevice)
        S = torch.tensor(graphs, device=architDevice)
        with torch.no_grad():
            thisaccels = archit(x, S)
        thisaccels = thisaccels.cpu().numpy()[:, -1, :, :]
        thisaccels[thisaccels > 3] = 3
        thisaccels[thisaccels < -3] = -3
        accels[:, -1, :, :] = thisaccels

        return ref_vel, est_vel, est_vels, biased_ref_vels, accels, positions

def agent_communication(wx, wy, n_agents, degree):
    positions = np.vstack((
        rng.uniform(0, wx, n_agents),
        rng.uniform(0, wy, n_agents)))
    distance = distance_matrix(positions.T, positions.T)
    network = copy.deepcopy(distance)
    for i in range(n_agents):
        re2 = heapq.nsmallest((degree + 1), distance[i, :])
        for j in range(n_agents):
            if distance[i, j] > re2[degree]:
                network[i, j] = 0
    network = network + np.transpose(network, axes=[1, 0])
    network[np.arange(0, n_agents), np.arange(0, n_agents)] = 0.
    network = (network > 0).astype(distance.dtype)
    W = np.linalg.eigvalsh(network)
    maxEigenvalue = np.max(np.real(W))
    normalized_network = network / maxEigenvalue

    return network, normalized_network


def agent_communication_pos(pos, n_agents, degree):
    distance = distance_matrix(pos.T, pos.T)
    network = copy.deepcopy(distance)
    for i in range(n_agents):
        re2 = heapq.nsmallest((degree + 1), distance[i, :])
        for j in range(n_agents):
            if distance[i, j] > re2[degree]:
                network[i, j] = 0
    network = network + np.transpose(network, axes=[1, 0])
    network[np.arange(0, n_agents), np.arange(0, n_agents)] = 0.
    network = (network > 0).astype(distance.dtype)
    W = np.linalg.eigvalsh(network)
    maxEigenvalue = np.max(np.real(W))
    normalized_network = network / maxEigenvalue

    return network, normalized_network


class DistributedController(ConsensusController):
    def __init__(self, adjacency, K=1, sample_time=0.1, energy_weight=1):

        self.nSamples = adjacency.shape[0]
        self.nAgents = adjacency.shape[1]
        self.K = K
        self.neighbor_network = np.zeros((self.nSamples, K, self.nAgents, self.nAgents))
        self.num_neighbor = np.zeros((self.nSamples, K, 2, self.nAgents))

        for i in range(self.nSamples):
            for k in range(1, (K + 1)):
                inter_matrix = np.linalg.matrix_power(adjacency[i, :, :], k)
                inter_matrix[inter_matrix >= 1] = 1
                self.num_neighbor[i, k - 1, 0, :] = np.sum(inter_matrix, 1)
                self.num_neighbor[i, k - 1, 1, :] = np.sum(inter_matrix, 1)
                self.neighbor_network[i, k - 1, :, :] = inter_matrix

        self.sample_time = sample_time
        self.energy_weight = energy_weight

    def __call__(self, est_vel, biased_vel, step):
        # find the averages along 0,K neighborhoods and average them
        biased_means = np.zeros((self.nSamples, self.K, 2, self.nAgents))

        if step < (self.K - 1):
            for k in range(step):
                biased_means[:, k, :, :] = np.matmul(biased_vel[:, step - k, :, :], self.neighbor_network[:, k, :, :])
                biased_means[:, k, :, :] = biased_means[:, k, :, :] / self.num_neighbor[:, k, :, :]
        else:
            for k in range(self.K):
                biased_means[:, k, :, :] = np.matmul(biased_vel[:, step - k, :, :], self.neighbor_network[:, k, :, :])
                biased_means[:, k, :, :] = biased_means[:, k, :, :] / self.num_neighbor[:, k, :, :]
        biased_mean = np.mean(biased_means, 1)
        return (biased_mean - est_vel[:, step, :, :]) / (self.sample_time * (1 + self.energy_weight))


def computeDifferences(u):
    # take input as: nSamples x tSamples x 2 x nAgents
    if len(u.shape) == 3:
        u = np.expand_dims(u, 1)
        hasTimeDim = False
    else:
        hasTimeDim = True
    nSamples = u.shape[0]
    tSamples = u.shape[1]
    nAgents = u.shape[3]
    uCol_x = u[:, :, 0, :].reshape((nSamples, tSamples, nAgents, 1))
    uRow_x = u[:, :, 0, :].reshape((nSamples, tSamples, 1, nAgents))
    uDiff_x = uCol_x - uRow_x
    uCol_y = u[:, :, 1, :].reshape((nSamples, tSamples, nAgents, 1))
    uRow_y = u[:, :, 1, :].reshape((nSamples, tSamples, 1, nAgents))
    uDiff_y = uCol_y - uRow_y
    uDistSq = uDiff_x ** 2 + uDiff_y ** 2
    uDiff_x = np.expand_dims(uDiff_x, 2)
    uDiff_y = np.expand_dims(uDiff_y, 2)
    uDiff = np.concatenate((uDiff_x, uDiff_y), 2)
    if not hasTimeDim:
        uDistSq = uDistSq.squeeze(1)
        uDiff = uDiff.squeeze(1)

    return uDiff, uDistSq


class OptimalControllerCollision(ConsensusController):
    def __init__(self, sample_time=0.1, energy_weight=1):
        self.sample_time = sample_time
        self.energy_weight = energy_weight

    def __call__(self, gama, d0, pos, est_vel, biased_vel, step):
        ijDiffPos, ijDistSq = networks.computeDifferences(pos[:, step, :, :])
        gammaMask = (ijDistSq < (gama ** 2)).astype(ijDiffPos.dtype)
        ijDiffPos = ijDiffPos * np.expand_dims(gammaMask, 1)
        ijDistSqInv = ijDistSq.copy()
        ijDistSqInv[ijDistSq < 1e-9] = 1.
        ijDistSqInv = 1. / ijDistSqInv
        ijDistSqInv[ijDistSq < 1e-9] = 0.
        ijDistSqInv = np.expand_dims(ijDistSqInv, 1)
        mean_biased_vel = np.mean(biased_vel[:, step, :, :], axis=2)
        mean_biased_vel = np.expand_dims(mean_biased_vel, 2)
        accel = -(est_vel[:, step, :, :] - mean_biased_vel) / (
                    self.sample_time * (1 + self.energy_weight)) + 1 / self.sample_time / 2 * 2 * np.sum(
            ijDiffPos * (d0 * d0 * (ijDistSqInv ** 2) + d0 * d0 * ijDistSqInv), axis=3)

        return accel


class OptimalControllerCollision(ConsensusController):
    def __init__(self, sample_time=0.1, energy_weight=1):
        self.sample_time = sample_time
        self.energy_weight = energy_weight

    def __call__(self, gama, d0, pos, est_vel, biased_vel, step):
        ijDiffPos, ijDistSq = networks.computeDifferences(pos[:, step, :, :])
        gammaMask = (ijDistSq < (gama ** 2)).astype(ijDiffPos.dtype)
        ijDiffPos = ijDiffPos * np.expand_dims(gammaMask, 1)
        ijDistSqInv = ijDistSq.copy()
        ijDistSqInv[ijDistSq < 1e-9] = 1.
        ijDistSqInv = 1. / ijDistSqInv
        ijDistSqInv[ijDistSq < 1e-9] = 0.
        ijDistSqInv = np.expand_dims(ijDistSqInv, 1)
        mean_biased_vel = np.mean(biased_vel[:, step, :, :], axis=2)
        mean_biased_vel = np.expand_dims(mean_biased_vel, 2)
        accel = -(est_vel[:, step, :, :] - mean_biased_vel) / (
                    self.sample_time * (1 + self.energy_weight)) + 1 / self.sample_time / 2 * 2 * np.sum(
            ijDiffPos * (d0 * d0 * (ijDistSqInv ** 2) + d0 * d0 * ijDistSqInv), axis=3)

        return accel