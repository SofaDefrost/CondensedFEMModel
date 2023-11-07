# -*- coding: utf-8 -*-
"""Config for the Finger Elasticity Design case.
"""

__authors__ = "emenager, tnavez"
__contact__ = "etienne.menager@inria.fr, tanguy.navez@inria.fr"
__version__ = "1.0.0"
__copyright__ = "(c) 2020, Inria"
__date__ = "Jan 12 2023"

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))

import numpy as np

from Libraries.BaseConfig import BaseConfig

class Config(BaseConfig):
    def __init__(self):
        super(Config,self).__init__("FingerElasticityParams")
        self.init_model_parameters()
        self.set_scene_config({"source": [-400.0, -50, 100],
                               "target": [30, -25, 100],
                               #"goalPos": [0, 10, -3.0*self.Length]
                               "goalPos": [0, 0, -3.0*self.Length]})

    #########################################################
    ### Methods for managing traveling in actuation space ###
    #########################################################
    def get_actuators_variables(self):
        return {"cable1": [10, 0, 20]}

    ######################################################
    ### Methods for managing traveling in design space ###
    ######################################################
    def init_model_parameters(self):
        # Geometric parameters
        self.Length = 40
        self.Height = 20
        self.JointHeight = 6
        self.Thickness = 17.5
        self.JointSlopeAngle = np.deg2rad(30)
        self.FixationWidth = 3

        self.OuterRadius = self.Thickness/2 + 6
        self.NBellows = 1
        self.BellowHeight = 8
        self.TeethRadius = self.Thickness/2
        self.WallThickness = 3
        self.CenterThickness = 1.5
        self.CavityCorkThickness = 3
        self.PlateauHeight = 3

        # Elasticity parameters
        self.PoissonRatio = 0.47 #0.4 - 0.499
        self.YoungsModulus = 5000 #1000 - 10000

        # Meshing parameters
        self.lc_finger = 7
        self.RefineAroundCavities = False

        # Mold parameters
        self.MoldWallThickness = 3
        self.MoldCoverTolerance = 0.1
        self.LengthMold = 3*self.Length + 2*self.MoldWallThickness
        self.LidHoleBorderThickness = 1
        self.LidHoleThickness = self.Thickness - 2*self.LidHoleBorderThickness
        self.LidHoleLength = 3*self.Length/5

        self.MoldHoleThickness = self.Thickness - 2*self.LidHoleBorderThickness
        self.MoldHoleLength = self.Length/2

        self.ThicknessMold = 2*self.OuterRadius + 2*self.MoldWallThickness
        self.LengthMold = 3*self.Length + 2*self.MoldWallThickness
        self.HeightMold = self.Height + self.FixationWidth + self.MoldWallThickness
        self.MoldHoleLidBorderThickness = 2

        # Cable
        self.CableRadius = 0.8
        self.CableDistance = 10
        self.CableHeight = 17.75


    def get_design_variables(self):
        return {
            "Poisson Ratio": [self.PoissonRatio, 0.40, 0.499],
            "Young's Modulus": [self.YoungsModulus, 1000, 10000]
        }

    #######################################
    ### Methods for managing simulation ###
    #######################################
    def get_n_dt(self):
        return 25

    def get_n_eq_dt(self):
        return 1

    def get_post_sim_n_eq_dt(self):
        return 80

    def get_trajectory(self):
        import numpy as np
        goals = []

        # Trajectory 1: finger closing on itself
        circle_center = np.array([0, 0, 0])
        circle_radius = 3.0*self.Length
        n_samples = 10
        for i in range(n_samples):
            goals.append(circle_center + np.array([0, circle_radius * np.cos(- (np.pi / 2) + (np.pi / 2)  * i / n_samples), circle_radius * np.sin(-(np.pi / 2) + (np.pi / 2)  * i / n_samples)]))

        # Trajectory 2: little circle with finger tip
        # circle_center = np.array([0, 0, -3.0*self.Length])
        # circle_radius = 20
        # n_samples = 10
        # for i in range(n_samples):
        #     goals.append(circle_center + np.array([circle_radius * np.cos(- (np.pi / 2) + (np.pi / 2)  * i / n_samples), circle_radius * np.sin((np.pi / 2) + (np.pi / 2)  * i / n_samples), 0]))

        return goals
