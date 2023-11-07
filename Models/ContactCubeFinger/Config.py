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
import numpy as np
class Config(BaseConfig):
    def __init__(self):
        super(Config,self).__init__("ContactFinger")
        self.scene_name = "ContactCubeFinger"
        self.nb_robot = 1
        self.set_scene_config({"source": [-400.0, -50, 100],
                               "target": [30, -25, 100],
                               "goalPos": [0, 35, -95.0, 0.0, 0.0, 0.0],
                               "contact_radius": 2,
                               "nb_contact_points": 3
})

    def get_actuators_variables(self):
        return {"cable1": [0, 0, 20],
                "cable2": [0, 0, 20]}

    def get_contacts_variables(self):
        x_comp = 30
        y_comp = 10
        z_comp = 20
        return {"contactP1_x": [x_comp, -40, 40],
                "contactP1_y": [y_comp, -20, 40],
                "contactP1_z": [z_comp, -2, 30],
                "contactP2_x": [x_comp, -40, 40],
                "contactP2_y": [y_comp, -20, 40],
                "contactP2_z": [z_comp, -2, 30],
                "contactP3_x": [x_comp, -40, 40],
                "contactP3_y": [y_comp, -20, 40],
                "contactP3_z": [z_comp, -2, 30]}

    def get_n_dt(self):
        return 5

    def get_n_eq_dt(self):
        return 1

    def get_trajectory(self):
        goal = [np.array([0, 35 + i*0.1, -95.0, 0.0, 0.0, 0.0]) for i in range (10)]
        return goal
