import numpy as np
import math

pi = math.pi
leg_offset = np.array([pi/3, pi/3, 0, 0, -pi/3, -pi/3]) + 0.173 * np.ones([6])


def calculate_a(cpg, cx,dist):
    p = np.tan(leg_offset + cx)
    a_cal = np.arctan(((-abs(dist) / cpg['s']) * (1 + p ** 2) * 2 + np.sqrt(
        (4 * abs(dist) ** 2) / (cpg['s'] ** 2) * (1 + p ** 2) ** 2 + 4 * p ** 2)) / (2 * p ** 2))
    a = np.reshape(a_cal, (1, 6))

    return a


