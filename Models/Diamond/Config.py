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
        super(Config,self).__init__("Diamond")
        self.set_scene_config({"source": [0, 400, 100],
                               "target": [0, 0, 100],
                               "goalPos": [0.0, 0.0, 125.0]})

    def get_actuators_variables(self):
        return {"north": [30, 0, 30],
                "west": [0, 0, 30],
                "south": [30, 0, 30],
                "east": [0, 0, 30]}

    def get_n_dt(self):
        return 10

    def get_n_eq_dt(self):
        return 10

    def get_trajectory(self):
        import numpy as np
        goals = []
        circle_center = np.array(self.scene_config["goalPos"]) + np.array([0.0, 0.0, 8.0])
        goals.append(circle_center)
        circle_radius = 20
        n_samples = 20
        for i in range(n_samples):
            goals.append(circle_center + np.array([circle_radius * np.cos(2 * np.pi * i / n_samples), circle_radius * np.sin(2 * np.pi * i / n_samples) , 0]))
        return goals
    
    
