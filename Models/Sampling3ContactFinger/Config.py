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
        self.scene_name = "Sampling3ContactFinger"
        self.nb_robot = 3
        self.is_direct_control_sampling = False
        self.set_scene_config({"source": [-400.0, -50, 100],
                               "target": [30, -25, 100],
                               "goalPos": [0., 35.0, -95.0, 0.0, 0.0, 0.0],
                               "contact_radius": 2,
                               "nb_contact_points": 3
        })

    def get_n_sampling_variables(self):
        return 6

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

    def get_inverse_variables(self):
        # return {"pos_cube_x": [0, -2, 2],
        #         "pos_cube_y": [35, 33, 37],
        #         "pos_cube_z": [-95, -95, -92],
        #         "ang_cube_x": [0, 0, 0],
        #         "ang_cube_y": [0, 0, 0],
        #         "ang_cube_z": [0, 0, 0],
        #         }

        return {"pos_cube_x": [0, -2, 2],
                "pos_cube_y": [35, 33, 37],
                "pos_cube_z": [-95, -95, -94],
                "ang_cube_x": [0, 0, 0],
                "ang_cube_y": [0, 0, 0],
                "ang_cube_z": [0, 0, 0],
                }
        

    def get_n_dt(self):
        return 5

    def get_n_eq_dt(self):
        return 1
    
    def get_trajectory(self):
        return []
