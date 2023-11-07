# -*- coding: utf-8 -*-
__authors__ = "emenager, tnavez"
__contact__ = "etienne.menager@inria.fr, tanguy.navez@inria.fr"
__version__ = "1.0.0"
__copyright__ = "(c) 2023, Inria"
__date__ = "Sept 20 2023"


import numpy as np
import math
def euler_to_quaternion(roll, pitch, yaw):
    """Compute the quaternion from the euler angles.
    Carreful: the angles are in radian.

    Parameters
    ----------
    roll, pitch, yaw: float
        Angles around x, y, z axis, in radian.

    Returns
    -------
        The associated quaternion.

    """
    roll, pitch, yaw = np.radians(roll), np.radians(pitch), np.radians(yaw)

    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

    return [qx, qy, qz, qw]

def quaternion_to_euler(x, y, z, w):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return [roll_x, pitch_y, yaw_z]


def pos_to_speed(X, X_prev, h):
    """Compute the speed of a list of positions using finite difference.
    ----------
    Parameters
    ----------
    X: list of numpy.arrays
        List of positions at the current time step
    X_prev: list of numpy.arrays
        List of positions at the previous time step
    h: float
        Time step in (s)
    ----------
    Outputs
    ----------
    V: list of numpy arrays
        List of computed velocities

    """
    V = [(X[i] - X_prev[i]) / h for i in range(len(X))]
    return V