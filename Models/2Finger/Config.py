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

import Mesh.Constants as Const

class Config(BaseConfig):
    def __init__(self):
        super(Config,self).__init__("Finger")
        self.scene_name = "2Finger"
        self.n_robot = 2
        self.set_scene_config({"source": [-400.0, -50, 100],
                               "target": [30, -25, 100],
                               "goalPos": [0, Const.Height/2, -3.0*Const.Length]})

    def get_actuators_variables(self):
        return {"cable1": [20, 0, 20],
                "cable2": [10, 0, 20]}

    def get_n_dt(self):
        return 30

    def get_n_eq_dt(self):
        return 2

    def get_trajectory(self):
        import numpy as np
        goals = []
        n_samples, max_x_pos = 5, 30

        init_point = np.array([- max_x_pos, Const.Height/2, -3.0*Const.Length])
        for i in range(2*n_samples):
            goals.append(init_point+ i*np.array([max_x_pos/n_samples, 0, 0]))

        return goals
