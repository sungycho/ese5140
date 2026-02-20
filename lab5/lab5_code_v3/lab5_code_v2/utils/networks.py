import numpy as np
from scipy.spatial import distance_matrix
import copy
import heapq
rng = np.random

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
