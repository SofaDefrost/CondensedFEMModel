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
        super(Config,self).__init__("ContactFinger")
        self.scene_name = "3ContactFinger"
        self.nb_robot = 3
        self.set_scene_config({"source": [-400.0, -50, 100],
                               "target": [30, -25, 100],
                               "goalPos": [0., 35.0, -95.0, 0.0, 0.0, 0.0],
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
        import numpy as np
        goals = []
        circle_center= np.array([0., 35.0, -95.0, 0, 0, 0])
        goals.append(circle_center)

        circle_radius = 1
        # circle_radius = 1.5
        n_samples = 20
        for i in range(n_samples):
            goals.append(circle_center + np.array([circle_radius * np.cos(2 * np.pi * i / n_samples), circle_radius * np.sin(2 * np.pi * i / n_samples),
                 0, 0, 0, 0]))
        return goals
