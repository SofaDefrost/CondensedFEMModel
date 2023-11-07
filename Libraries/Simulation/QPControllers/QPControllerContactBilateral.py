# -*- coding: utf-8 -*-
"""Base controller to interact with the Sofa scene.
"""

__authors__ = "emenager, tnavez"
__contact__ = "etienne.menager@inria.fr, tanguy.navez@inria.fr"
__version__ = "1.0.0"
__copyright__ = "(c) 2020, Inria"
__date__ = "Jul 19 2022"

import Sofa
import SofaRuntime
from Sofa import SofaConstraintSolver
import sys
import pathlib
import importlib
import json
import torch
import numpy as np
import copy
import math
import pickle

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()) + "/../../")
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))

from QPControllerContact import QPController as QPControllerContact
from QPControllerContact import compute_mechanical_matrices_from_MLP

from QPproblem import init_QP, solve_QP
from database import query_simulation_data, query_sampling_stats, query_recover_W_dfree_from_s_a
from utils import ask_user_sampling_stat_input, create_bloc
from geometry import euler_to_quaternion, quaternion_to_euler

class QPController(QPControllerContact):
    def __init__(self, *args, **kwargs):
        """Classical initialization of a python class.

        Note:
        ----
            The kwargs argument must containe:
                - root: the root of the SOFA scene.
                - network_name: the model's name we want to use in {"MLP", "GNN"}
                - config: the config of the model.
                - type_use: use learned matrices, simulated one or interpolate between matrices.
                - intermediate_goals: trajectory shaped as a list of list of goals to successively reach during simulation
        """
        assert kwargs.get("name") == "QPController"
        QPControllerContact.__init__(self, *args, **kwargs)

    ########### For the bilateral case some changes to respect the scene / the configuration

    def init_model(self):
        utils_lib = importlib.import_module("Libraries.utils")
        model, _, _, dataloader, _, n_constraint, self.data_scaling, _, _, _, best_model_link, _ = utils_lib.init_network(self.network_name, self.config)

        self.W_0 = dataloader.dataset.X[0][0]
        self.dfree_0 = dataloader.dataset.X[0][1]
        return model, n_constraint, best_model_link

    def compute_B_from_dfree_c(self, X_cube, lambda_c, D, Jc, Wcc, dfree_c, xc_cube):
        Wcc_inv = np.linalg.inv(Wcc)
        JcWcc_inv = np.matmul(Jc.T, Wcc_inv)
        inter_1 = np.matmul(D, JcWcc_inv)
        inter_2 = np.matmul(inter_1, dfree_c + np.matmul(Jc, X_cube) - xc_cube)
        B = inter_2

        return B

    def recover_from_simulation(self, nb_robot, n_constraint, n_act_constraint, W_t, dfree_t):
        return compute_mechanical_matrices_from_simulation(nb_robot, n_constraint, n_act_constraint, W_t, dfree_t)

    def get_cube_pos(self):
        cube_pos = self.root.Simulation.Cube.MechanicalObject.position.value[0]
        cube_trans = cube_pos[:3].tolist()
        [x, y, z, w] = cube_pos[3:].tolist()

        #Convert quaternion to euler
        [roll_x, pitch_y, yaw_z] = quaternion_to_euler(x, y, z, w)
        return np.array(cube_trans + [roll_x, pitch_y, yaw_z])


    def update_save_data(self):
        goal_pos = self.config.scene_config["goalPos"]
        effector_pos = self.root.Simulation.Cube.MechanicalObject.position.value[0]

        self.save_data["pos_effector"]["x"].append(effector_pos[0])
        self.save_data["pos_effector"]["y"].append(effector_pos[1])
        self.save_data["pos_effector"]["z"].append(effector_pos[2])

        self.save_data["pos_goal"]["x"].append(goal_pos[0])
        self.save_data["pos_goal"]["y"].append(goal_pos[1])
        self.save_data["pos_goal"]["z"].append(goal_pos[2])

        # Registering effector poses for debug purpose
        s_a = self.root.Controller.get_actuators_state()
        for finger_actuation_state in s_a:
            s_c = finger_actuation_state[self.n_act_constraint:]
            for i in range(3):
                self.save_data_finger_effectors["pos_effector"]["x"].append(s_c[3 * i])
                self.save_data_finger_effectors["pos_effector"]["y"].append(s_c[3 * i + 1])
                self.save_data_finger_effectors["pos_effector"]["z"].append(s_c[3 * i + 2])


    def onSolveConstraintSystemEndEvent(self, event):
        #To avoid computation from BaseQPController
        pass

    ####### Implement solve_QP and apply action to control the displacement of the actuators

    def solve_QP_problem(self, data_QP_matrices):
        # Suppose we have rigid bar
        data_QP_matrices["l"] = np.zeros(data_QP_matrices["l"].shape) - np.infty
        new_lambda_a = solve_QP(self.solver, data_QP_matrices["H"], data_QP_matrices["g"], data_QP_matrices["C"],
                                data_QP_matrices["l"], data_QP_matrices["u"], data_QP_matrices["lb"], data_QP_matrices["ub"],
                                is_init=(self.step != self.waiting_time))

        factor = 1
        self.lambda_a = self.lambda_a + factor * (new_lambda_a - self.lambda_a)

        # Compute new object position
        new_X_cube = self.compute_X(data_QP_matrices["A"], data_QP_matrices["B"], self.lambda_a)


        # Compute new contact forces
        self.lambda_c = self.compute_lambda_c_from_dfree_c(data_QP_matrices["X_cube"], new_X_cube, data_QP_matrices["Wcc"],
                                                        data_QP_matrices["Wca"], data_QP_matrices["Jc"], self.lambda_a,
                                                          data_QP_matrices["xc_cube"], data_QP_matrices["dfree_c"])


        data_actuation = {}
        data_actuation["delta_a"] = np.matmul(data_QP_matrices["Waa"], self.lambda_a) + np.matmul(data_QP_matrices["Wca"].T, self.lambda_c) + data_QP_matrices["dfree_a"]
        return data_actuation

    def apply_actuation(self, data_actuation):
        for i, actuator in enumerate(self.list_actuator):
            actuator.value.value = [data_actuation["delta_a"][i]]

    ####### The bilateral case works with onAnimateBeginEvents
    def onAnimateBeginEvent(self, event):
        if self.step == self.waiting_time:
            self.init_actuation()

        if self.step < self.waiting_time:
            print("[INFO]  >>  Waiting time {}/{}".format(self.step, self.waiting_time))
        else:
            data_matrices = self.recover_matrices()
            data_QP_matrices = self.build_QP_matrices(data_matrices)
            data_actuation = self.solve_QP_problem(data_QP_matrices)
            self.apply_actuation(data_actuation)


            # Update goal position if managing trajectory
            if self.simu_results:
                eq_counter = self.step % self.waiting_eq_dt
                if eq_counter == 0:
                    self.update_goal_position()
        self.step += 1








def compute_mechanical_matrices_from_simulation(n_finger, n_constraint, n_act_constraint, W_t, dfree_t):
    """
    Rebuild QP matrices for QP solving

    W = |Waa^1  Wac^1  0      0       0       0    |
        |Wca^1  Wcc^1  0      0       0       0    |
        |0      0      Waa^2  Wac^2   0       0    |
        |0      0      Wca^2  Wcc^2   0       0    |
        |0      0      0      0       Waa^3   Wac^3|
        |0      0      0      0       Wca^3   Wcc^3|


    Parameters
    ----------
    n_constraint: int
        Total number of constraints
    n_act_constraint: int
        Number of constraints for actuators
    W_t: numpy array
        Compliance matrice projected in constraint space
    dfree_t: numpy array
        Displacement in free configuration

    Outputs
    ----------
    list_Wcc: list of array
        List containing W_cc^i for i in [1, 2, 3]
    list_Wca: list of array
        List containing W_ca^i for i in [1, 2, 3]
    list_Waa: list of array
        List containing W_aa^i for i in [1, 2, 3]
    list_delta_a_free: list of array
        List containing delta_a_free^i for i in [1, 2, 3]
    list_delta_c_free: list of array
        List containing delta_c_free^i for i in [1, 2, 3]
    """
    list_Wcc, list_Wca, list_Waa, list_delta_a_free, list_delta_c_free = [], [], [], [], []
    n_contact = n_constraint - n_act_constraint

    slide = 3*n_contact
    for i in range(n_finger):
        Waa_i = W_t[slide+i * n_constraint: slide+i * n_constraint + n_act_constraint, slide+i * n_constraint:slide+ i * n_constraint + n_act_constraint]
        Wca_i = W_t[slide+i * n_constraint + n_act_constraint :slide+(i + 1) * n_constraint,
                slide+i * n_constraint: slide+i * n_constraint + n_act_constraint]
        Wcc_i = W_t[slide+i * n_constraint + n_act_constraint: slide+(i + 1) * n_constraint,
                slide+i * n_constraint + n_act_constraint: slide+(i + 1) * n_constraint]


        list_Wcc.append(Wcc_i)
        list_Wca.append(Wca_i)
        list_Waa.append(Waa_i)

        dfree_a_i = dfree_t[slide+i * n_constraint: slide+i * n_constraint + n_act_constraint]
        list_delta_a_free.append(dfree_a_i)

        dfree_c_i = dfree_t[slide+i * n_constraint + n_act_constraint: slide+(i + 1) * n_constraint]
        list_delta_c_free.append(dfree_c_i)

    return list_Wcc, list_Wca, list_Waa, list_delta_a_free, list_delta_c_free


