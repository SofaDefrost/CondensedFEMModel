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
        super(Config,self).__init__("FingerFine")
        self.set_scene_config({"source": [-400.0, -50, 100],
                               "target": [30, -25, 100],
                               "goalPos": [0, 10, -3.0*Const.Length]})

    def get_actuators_variables(self):
        return {"cable1": [20, 0, 20],
                "cable2": [20, 0, 20]}

    def get_n_dt(self):
        return 15

    def get_n_eq_dt(self):
        return 1
    
    def get_trajectory(self):
        import numpy as np
        goals = []
        
        # Trajectory 1: finger closing on itself
        circle_center = np.array([0, 0, 0])
        circle_radius = 3.0*Const.Length
        n_samples = 40
        for i in range(n_samples):
            goals.append(circle_center + np.array([0, circle_radius * np.cos(- (np.pi / 2) + (np.pi / 2)  * i / n_samples), circle_radius * np.sin(-(np.pi / 2) + (np.pi / 2)  * i / n_samples)]))
            
        # Trajectory 2: little circle with finger tip
        # circle_center = np.array([0, 0, -3.0*Const.Length])
        # circle_radius = 20
        # n_samples = 10
        # for i in range(n_samples):
        #     goals.append(circle_center + np.array([circle_radius * np.cos(- (np.pi / 2) + (np.pi / 2)  * i / n_samples), circle_radius * np.sin((np.pi / 2) + (np.pi / 2)  * i / n_samples), 0]))
        
        return goals
        
