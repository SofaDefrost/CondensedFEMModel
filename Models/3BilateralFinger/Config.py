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
        super(Config,self).__init__("BilateralFinger")
        self.scene_name = "3BilateralFinger"
        self.nb_robot = 3
        self.set_scene_config({"source": [-400.0, -50, 100],
                               "target": [30, -25, 100],
                               "goalPos": [0., 35.0, -95.0, 0.0, 0.0, 0.0],
                               "contact_radius": 2,
                               "nb_contact_points": 3,
                               "is_force": False
})

    def get_actuators_variables(self):
        return {"cable1": [-4, 0, 20],
                "cable2": [-4, 0, 20]}

    def get_contacts_variables(self):
        x_comp = 0
        y_comp = 0
        z_comp = 0
        return {"contactP1_x": [x_comp, -40, 40],
                "contactP1_y": [y_comp, -20, 40],
                "contactP1_z": [z_comp, -2, 30],
                "contactP2_x": [x_comp, -40, 40],
                "contactP2_y": [y_comp, -20, 40],
                "contactP2_z": [z_comp, -2, 30],
                "contactP3_x": [x_comp, -40, 40],
                "contactP3_y": [y_comp, -20, 40],
                "contactP3_z": [z_comp, -2, 30]}

        # return {"contactP1_x": [x_comp, -40, 40],
        #         "contactP1_y": [y_comp, -20, 40],
        #         "contactP1_z": [z_comp, -2, 30]}


    def get_n_dt(self):
        return 10

    def get_n_eq_dt(self):
        return 4
    
    def get_trajectory(self):
        import numpy as np        
        goals = []
        
        # 2D Circle trajectory
        circle_center= np.array([0., 35.0, -95.0, 0, 0, 0])
        goals.append(circle_center)

        circle_radius = 1.5
        n_samples = 20
        for i in range(n_samples):
            goals.append(circle_center + np.array([circle_radius * np.cos(2 * np.pi * i / n_samples), circle_radius * np.sin(2 * np.pi * i / n_samples),
                 0, 0, 0, 0]))
              
        # # 3D spiral trajectory
        # center= np.array([0., 35.0, -95.0, 0, 0, 0])
        # radius = 1.5
        # n_samples = 30
        # height_increment = 1 / n_samples
        # additional_height = 0.0
        # angle_increment = 2 * 6.28319 / n_samples
        # additional_angle = 0.0
        #
        # for i in range(n_samples):
        #     x = center[0] + radius * np.cos(additional_angle)
        #     y = center[1] + radius * np.sin(additional_angle)
        #     z = center[2] + additional_height
        #     goals.append(np.array([x,y,z, 0, 0, 0]))
        #
        #     additional_height += height_increment
        #     additional_angle += angle_increment
            
        return goals

        