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

import numpy as np

import Mesh.Constants as Const

class Config(BaseConfig):
    def __init__(self):
        super(Config,self).__init__("ContactFinger")
        self.set_scene_config({"source": [-400.0, -50, 100],
                               "target": [30, -25, 100],
                               "goalPos": [0, 0, -3.0*Const.Length],
                               "contact_radius": 2,
                               "nb_contact_points": 3})

    def get_actuators_variables(self):
        return {"cable1": [0, -2, 20],
                "cable2": [0, -2, 20]}
                
    def get_contacts_variables(self):
        # # Forces applied directly on contact points
        # x_comp = 30
        # y_comp = 10
        # z_comp = 20
        # return {"contactP1_x": [x_comp, -40, 40],
        #         "contactP1_y": [y_comp, -20, 40],
        #         "contactP1_z": [z_comp, -2, 30],
        #         "contactP2_x": [x_comp, -40, 40],
        #         "contactP2_y": [y_comp, -20, 40],
        #         "contactP2_z": [z_comp, -2, 30],
        #         "contactP3_x": [x_comp, -40, 40],
        #         "contactP3_y": [y_comp, -20, 40],
        #         "contactP3_z": [z_comp, -2, 30]}

        # Testing to sample displacement of the contact plane
        sampling_contact_variables = self.get_sampling_contacts_variables()
        values = [sampling_contact_variables[var_name][0] for var_name in sampling_contact_variables]
        contact_point_values = self.plane_to_contacts(values)
        return {"contactP1_x": [contact_point_values[0], -40, 40],
                "contactP1_y": [contact_point_values[1], -20, 40],
                "contactP1_z": [contact_point_values[2], -2, 30],
                "contactP2_x": [contact_point_values[3], -40, 40],
                "contactP2_y": [contact_point_values[4], -20, 40],
                "contactP2_z": [contact_point_values[5], -2, 30],
                "contactP3_x": [contact_point_values[6], -40, 40],
                "contactP3_y": [contact_point_values[7], -20, 40],
                "contactP3_z": [contact_point_values[8], -2, 30]}

    def get_n_dt(self):
        return 20

    def get_n_eq_dt(self):
        return 1
    
    def get_trajectory(self):
        import numpy as np
        goals = []
        
        # Trajectory 1: finger closing on itself
        circle_center = np.array([0, 0, 0])
        circle_radius = 3.0*Const.Length
        n_samples = 10
        for i in range(n_samples):
            goals.append(circle_center + np.array([0, circle_radius * np.cos(- (np.pi / 2) + (np.pi / 2)  * i / n_samples), circle_radius * np.sin(-(np.pi / 2) + (np.pi / 2)  * i / n_samples)]))
            
        # Trajectory 2: little circle with finger tip
        # circle_center = np.array([0, 0, -3.0*Const.Length])
        # circle_radius = 20
        # n_samples = 10
        # for i in range(n_samples):
        #     goals.append(circle_center + np.array([circle_radius * np.cos(- (np.pi / 2) + (np.pi / 2)  * i / n_samples), circle_radius * np.sin((np.pi / 2) + (np.pi / 2)  * i / n_samples), 0]))
        
        return goals
    
    ### Methods specific to sampling in the contact space
    def get_sampling_contacts_variables(self):
    # We sample in the space of translation and rotations of a plane containing the contact points
        return {"contactT_x": [0, -30, 30],
                "contactT_y": [0, -20, 30],
                "contactT_z": [0, -5, 20],
                "contactR_x": [0, -40, 40],
                "contactR_y": [0, -40, 40],
                "contactR_z": [0, -40, 40]}

    def get_n_sampling_variables(self):
        return len(self.get_actuators_variables()) + len(self.get_sampling_contacts_variables())

    def interpolate_variables(self, normalized_values, var_type = "actuation"):
        if var_type == "contact":
            variables = list(self.get_sampling_contacts_variables().values())
        else: #Actuation by default
            variables = list(self.get_actuators_variables().values())
        values = [normalized_values[i] * (variables[i][2] - variables[i][1]) + variables[i][1] for i in range(len(normalized_values))]
        
        # Sample translation and rotation of the contact plane
        if var_type == "contact":
            values = self.plane_to_contacts(values)

        return values
    
    def plane_to_contacts(self, values):
        # Init contact pos
        position_center = np.array([0, 0, 0])
        ang = np.pi/self.scene_config["nb_contact_points"]
        pos_contacts = []
        for i in range(self.scene_config["nb_contact_points"]):
            pos_contacts.append(np.array([position_center[0]+self.scene_config["contact_radius"]*np.sin(2*i*ang), position_center[1], position_center[2]+self.scene_config["contact_radius"]*np.cos(2*i*ang)]))
        # Translate and rotate positions
        T = np.array([values[0], values[1], values[2]])
        R = self.euler_to_rotation_matrice(values[3], values[4], values[5])
        for i in range(self.scene_config["nb_contact_points"]):
            pos_contacts[i] = T - pos_contacts[i] + np.matmul(R, pos_contacts[i])
        # Final values
        values = []
        for i in range(self.scene_config["nb_contact_points"]):
            values += pos_contacts[i].tolist()
        return values


    def euler_to_rotation_matrice(self, roll, pitch, yaw):
        c1 = np.cos(roll * np.pi / 180)
        s1 = np.sin(roll * np.pi / 180)
        c2 = np.cos(pitch * np.pi / 180)
        s2 = np.sin(pitch * np.pi / 180)
        c3 = np.cos(yaw * np.pi / 180)
        s3 = np.sin(yaw * np.pi / 180)
        rot_matrix=np.array([[c2*c3, -c2*s3, s2],
                         [c1*s3+c3*s1*s2, c1*c3-s1*s2*s3, -c2*s1],
                         [s1*s3-c1*c3*s2, c3*s1+c1*s2*s3, c1*c2]])
        return rot_matrix



        
        
        
