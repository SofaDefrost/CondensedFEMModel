# -*- coding: utf-8 -*-
"""Config for the Trunk.
"""

__authors__ = "emenager, tnavez"
__contact__ = "etienne.menager@inria.fr, tanguy.navez@inria.fr"
__version__ = "1.0.0"
__copyright__ = "(c) 2020, Inria"
__date__ = "Jun 29 2022"

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))

from Libraries.BaseConfig import BaseConfig

class Config(BaseConfig):
    def __init__(self):
        super(Config,self).__init__("Trunk")
        self.set_scene_config({"source": [-400.0, -50, 100],
                               "target": [30, -25, 100],
                               "goalPos": [0, 0, 195]})

    def get_actuators_variables(self):
        return {"cableS0": [0, 0, 13],
                "cableS1": [0, 0, 13],
                "cableS2": [13, 0, 13],
                "cableS3": [13, 0, 13],
                "cableL0": [0, 0, 13],
                "cableL1": [0, 0, 13],
                "cableL2": [13, 0, 13],
                "cableL3": [0, 0, 13]}

    def get_n_dt(self):
        return 70
    
    def get_n_eq_dt(self):
        return 60

    def get_trajectory(self):
        import numpy as np
        import math
        goals = []
        r = self.scene_config["goalPos"][2]
        n_samples = 10

        # Sample on Fibonnacci Demi Sphere
        N = n_samples * 2
        phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians
        for i in range(N):
            y = 1 - (i / float(N - 1)) * 2  # y goes from 1 to -1
            radius = math.sqrt(1 - y * y)  # radius at y
            theta = phi * i  # golden angle increment
            x = math.cos(theta) * radius
            z = math.sin(theta) * radius
            goals.append(np.array([r *x, r *y, r *z]))

        goals = [p for p in goals if p[2]>0]
        return goals