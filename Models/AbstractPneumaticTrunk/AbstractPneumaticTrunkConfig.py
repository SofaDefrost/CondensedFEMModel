# -*- coding: utf-8 -*-
"""Config for the AbstractPneumaticTrunk.py
"""

__authors__ = "emenager, tnavez"
__contact__ = "etienne.menager@inria.fr, tanguy.navez@inria.fr"
__version__ = "1.0.0"
__copyright__ = "(c) 2020, Inria"
__date__ = "March 23 2023"

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()) + "/../")
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))

from Libraries.BaseConfig import BaseConfig

class AbstractPneumaticTrunkConfig(BaseConfig):
    def __init__(self, name):
        super(AbstractPneumaticTrunkConfig, self).__init__(name)
        self.set_scene_config({"goalPos": [100, 2, 3.46, 100, 2, -3.46, 100, -4, 0],
                               "rigidGoalPos": [110, 0, 0, 0, 0, 0],
                               "use3Effectors": True, # Else we use only 1 effector
                               "actuatorStateType": "volume",  #Specify if we use pressure or volume as actuator state
                               "name": name})
        self.useRigidGoal = True


    def get_actuators_variables(self):
        return {"cavity1": [10, 0, 30],
                "cavity2": [0, 0, 30],
                "cavity3": [3, 0, 30],
                "cavity4": [0, 0, 30],
                "cavity5": [5, 0, 30],
                "cavity6": [0, 0, 30],}
        
    def get_n_eq_dt(self):
        return 1
    def get_n_dt(self):
        return 20
    def get_post_sim_n_eq_dt(self):
        return 25 # 50

    def get_trajectory(self):
        import numpy as np
        goals = []

        # A little circle
        circle_center = np.array([110, 0, 0, 0, 0, 0])
        circle_radius = 20
        n_samples = 15
        for i in range(n_samples):
            goals.append(circle_center + np.array([0, circle_radius * np.cos(2 * np.pi * i / n_samples), 0, 0, 0, 0]))

        # goals = np.array([[90, 0, 0, 0, 0, 0] for _ in range(20)])

        return goals

