import numpy as np
from CPG.computeOffsets import computeOffsets
from Tools.constrainSE3 import *
from CPG.limitValue import *
from CPG.groundIK import *
import copy
import math
import time
from matplotlib import pyplot as plt

pi = np.pi


def CPG(cpg, t, dt):
    np.set_printoptions(precision=5)

    shoulders1 = range(0, 18, 3)  # joint IDs of the shoulders
    shoulders2 = range(1, 18, 3)  # joint IDs of the second shoulder joints
    elbows = range(2, 3, 18)  # joint IDs of the elbow joints

    # dynOffsetError = copy.deepcopy(np.zeros([3, 6]))

    # if t > cpg['initLength']:
    #     [dynOffsetError, cpg] = computeOffsets(cpg, t, dt)

    # Scale
    # dynOffsetError = dynOffsetError * cpg['scaling']*0

    # CPG Implmentation/Equations
    gamma_x = 40  # forcing to limit cycle (larger = stronger)
    gamma_y = 40
    # lambdaa_x = 0.1  # coupling strength (larger = weaker)
    lambdaa_y = 0.1  # coupling strength
    cpg['d'] = 2  ##define the shape of the curvature
    cpg['w'] = 6  ##speed of the gait cycle
    # cpg['gamma'] = gamma
    # print(cpg['b'])
    # print(cpg['a'])

    if cpg['move']:

        dx = -1 * -1 * ((cpg['a'] * cpg['b']) / 2) * cpg['w'] * 2 * (
                (cpg['y'][t, :] - cpg['cy']) / (cpg['b'] ** 2)) + gamma_x * (
                     1 - (((cpg['x'][t, :] - cpg['cx']) / cpg['a']) ** 2) - (
                     (cpg['y'][t, :] - cpg['cy']) / cpg['b']) ** 2) * 2 * (
                     (cpg['x'][t, :] - cpg['cx']) / (cpg['a'] ** 2))
        # + (np.dot(lambdaa_x * cpg['K'], (cpg['x'][t, :] - cpg['cx'])))
        dy = -1 * ((cpg['a'] * cpg['b']) / 2) * cpg['w'] * 2 * (
                (cpg['x'][t, :] - cpg['cx']) / (cpg['a'] ** 2)) + gamma_y * (
                     1 - (((cpg['x'][t, :] - cpg['cx']) / cpg['a']) ** 2) - (
                     (cpg['y'][t, :] - cpg['cy']) / cpg['b']) ** 2) * 2 * (
                     (cpg['y'][t, :] - cpg['cy']) / (cpg['b'] ** 2)) + (
                 np.dot(lambdaa_y * cpg['K'], (cpg['y'][t, :] - cpg['cy'])))

        for value in range(6):
            truth = (cpg['CPGStanceDelta'].flatten())[value]
        truther = False


    else:
        dx = 0
        dy = 0
        truther = True
        dx_const = 0
        updStab = np.logical_or(cpg['CPGStance'], np.array([False, False, False, False, False, False]))

    # Integrate dx & dy to produce joint commands
    cpg['x'][t + 1, :] = cpg['x'][t, :] + dx * dt  # + cpg['dynOffsetInc'][0,:]
    cpg['y'][t + 1, :] = cpg['y'][t, :] + dy * dt  # + cpg['dynOffsetInc'][1,:]

    # Command CPG-generated values to joints
    yOut = cpg['y'][t + 1, :]
    xOut = np.zeros([1, 6])

    SignBack = -1 if (cpg['direction'] == 'backwards') else 1
    SignLeft = -1 if (cpg['direction'] == 'left') else 1
    SignRight = -1 if (cpg['direction'] == 'right') else 1

    for value in range(6):
        if value % 2 == 0:
            xOut[:, value] = SignLeft * SignBack * cpg['x'][t + 1, value]
        else:
            xOut[:, value] = SignRight * SignBack * cpg['x'][t + 1, value]

    cpg['legs'][0, 0:18:3] = limitValue((cpg['nomOffset'][0, :] + cpg['shouldersCorr'] * xOut), pi / 2 * cpg['scaling'])
    cpg['legs'][0, 1:19:3] = cpg['shouldersCorr'] * np.maximum(0, yOut) + cpg['nomOffset'][1, :]
    # JOINT 3 - FOR WALKING TRIALS
    cpg['legs'][0, 0:18:3] = cpg['legs'][0, 0:18:3] / cpg['scaling']
    cpg['legs'][0, 1:19:3] = cpg['legs'][0, 1:19:3] / cpg['scaling']

    I = (np.logical_or(cpg['CPGStance'], dy < 0))  # the leg on the ground

    indicies = []  # the legs are falling and on the ground
    for index in range(6):
        if truther:  # moving : truther = false
            truth = I[index]
        else:
            truth = I[0, index]
        if truth:
            indicies.append(index)

    Leglength = 0.325
    z = np.array([-math.pi / 2, math.pi / 2, -math.pi / 2, math.pi / 2, -math.pi / 2, math.pi / 2])

    if cpg['t'] > (cpg['initLength'] + 100) and cpg['direction'] == 'forward':

        dist = cpg['foothold_offsetY']
        angs = groundIK(cpg, dist, indicies)
        for index in indicies:
            cpg['legs'][0, 2 + index * 3] = angs[2 + index * 3]  # +(cpg['dynOffset'][2,index]/cpg['scaling'])

    cpg['legs'] = np.reshape(cpg['legs'][0:18], [1, 18])

    if cpg['t'] > (cpg['initLength'] + 100):
        positions = cpg['xmk'].getLegPositions(cpg['legs'])
    else:
        positions = np.array([[0.50588, 0.63498, 0.1406, -0.00552, -0.47454, -0.54181],
                              [0.2559, -0.25935, 0.60987, -0.5175, 0.31018, -0.42041],
                              [-0.2249, -0.00234, -0.02791, -0.2249, -0.2249, -0.00273]])

    return cpg, positions
