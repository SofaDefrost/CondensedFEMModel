# -*- coding: utf-8 -*-
"""Base controller to interact with the Sofa scene.
"""

__authors__ = "emenager, tnavez"
__contact__ = "etienne.menager@inria.fr, tanguy.navez@inria.fr"
__version__ = "1.0.0"
__copyright__ = "(c) 2020, Inria"
__date__ = "Jul 19 2022"

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../../")
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))

from QPControllerSimple import *
from QPControllerSimple import QPController as QPControllerSimple
from geometry import pos_to_speed

class QPController(QPControllerSimple):
    def __init__(self, *args, **kwargs):
        """Classical initialization of a python class.

        Note:
        ----
            The kwargs argument must containe:
                - root: the root of the SOFA scene.
                - network_name: the model's name we want to use in {"MLP"}
                - config: the config of the model.
                - type_use: use learned matrices, simulated one or interpolate between matrices.
                - intermediate_goals: trajectory shaped as a list of list of goals to successively reach during simulation
        """
        assert kwargs.get("name") == "QPController"
        QPControllerSimple.__init__(self, *args, **kwargs)

        self.list_X_e_simu = []
        self.list_V_e_simu = []
        self.list_delta_e = []
        self.list_V_e = [] 


    def build_QP_matrices(self, data_matrices):
        
        # Compute matrices relative to QP problem
        data_QP_matrices = QPControllerSimple.build_QP_matrices(self, data_matrices)

        # Add matrices relative to mechanical state
        data_QP_matrices["Wea"] = data_matrices["Wea"]
        data_QP_matrices["dfree_e"] = data_matrices["dfree_e"]

        return data_QP_matrices


    def solve_QP_problem(self, data_QP_matrices):
        
        # Solve and compute new actuation to apply  
        data_actuation =  QPControllerSimple.solve_QP_problem(self, data_QP_matrices)

        # Store previous X_e computed in simulation
        effector_pos = np.array(self.root.Controller.get_effectors_positions()[0].tolist()[0])
        self.list_X_e_simu.append(effector_pos)
        
        # Compute next delta_e 
        delta_e = np.dot(data_QP_matrices["Wea"], data_actuation["lambda_a"])  + data_QP_matrices["dfree_e"]
        self.list_delta_e.append(delta_e)

        # Compute V_e for current iteration
        if len(self.list_X_e_simu) >= 3:
            V_e_simu = (self.list_X_e_simu[-1] - self.list_X_e_simu[-2]) / self.root.dt.value
            V_e = (self.list_delta_e[-2] -  self.list_delta_e[-3]) / self.root.dt.value
            print("V_e_simu:", V_e_simu)
            print("V_e:", V_e)


        return data_actuation
