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
import pickle
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../../")
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))

from Libraries.geometry import euler_to_quaternion

class BaseQPController(Sofa.Core.Controller):
    def __init__(self, monitoring_time = False, *args, **kwargs):
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
        Sofa.Core.Controller.__init__(self, *args, **kwargs)

        self.root = kwargs.get("root")
        self.network_name = kwargs.get("network_name")
        self.config = kwargs.get("config")
        self.waiting_time = self.config.get_n_eq_dt()
        self.waiting_eq_dt = self.config.get_n_dt()
        self.type_use = kwargs.get("type_use")
        self.normalization_method = self.config.config_network["data_normalization"]
        self.monitoring_time = monitoring_time

        if self.type_use == "learned":
            self.model, self.n_constraint, best_model_link = self.init_model()
            self.load_model(best_model_link)
            self.model.eval()

        self.n_act_constraint = len(self.config.get_actuators_variables())
        self.init_QP()

        self.step = 0
        self.real_time_computation = 0

        # Load min / max actuation displacement states
        actuations_disp_bounds = list(self.config.get_actuators_variables().values())
        self.min_s_a = [actuations_disp_bounds[i][1] for i in range(len(actuations_disp_bounds))]
        self.max_s_a = [actuations_disp_bounds[i][2] for i in range(len(actuations_disp_bounds))]

        # Manage a trajectory consisting in several intermediate goals
        self.intermediate_goals = kwargs.get("intermediate_goals")
        self.simu_results = False
        if len(self.intermediate_goals) > 0:
            self.simu_results = True
        self.save_data = {"pos_effector": {"x": [], "y": [], "z": []}, "pos_goal": {"x": [], "y": [], "z": []}}

        self.init_correction()

    def init_correction(self):
        pass

    def apply_correction(self, dfree, Wea):
        return dfree

    def reinit_correction(self):
        pass

    def init_model(self):
        utils_lib = importlib.import_module("Libraries.utils")
        model, _, _, _, _, n_constraint, self.data_scaling, _, _, _, best_model_link, _ = utils_lib.init_network(self.network_name, self.config)
        return model, n_constraint, best_model_link

    def init_QP(self):
        print("[ERROR] >> You have to implement init_QP.")
        exit(1)
    def recover_matrices(self):
        print("[ERROR] >> You have to implement recover_matrices.")
        print(">> Output: dictionnary of mechanical matrices")
        exit(1)
    def build_QP_matrices(self, data_matrices):
        print("[ERROR] >> You have to implement compute_QP_matrices.")
        print(">> Input: dictionnary of mechanical matrices")
        print(">> Output: dictionnary of QP matrices")
        exit(1)
    def solve_QP_problem(self, data_QP_matrices):
        print("[ERROR] >> You have to implement solve_QP_problem.")
        print(">> Input: dictionnary of QP matrices")
        print(">> Output: actuation")
        exit(1)

    def init_actuation(self):
        print("[ERROR] >> You have to implement init_actuation.")
        exit(1)
    def apply_actuation(self, data_actuation):
        print("[ERROR] >> You have to implement apply_actuation.")
        print(">> Input: actuation")
        exit(1)

    def load_model(self, best_model_link):
        print("best_model_link:", best_model_link)
        if pathlib.Path.exists(best_model_link):
            checkpoint = torch.load(best_model_link)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            best_epoch = checkpoint['epoch']
            best_test_loss = checkpoint['loss']
            print(">>   Reload the best model from epoch {} with test loss {}".format(best_epoch, best_test_loss))
        else:
            print("[ERROR]  >>  No file to load the best model.")
            exit(1)


    def onSolveConstraintSystemEndEvent(self, event):
        if int(self.step) == 0:
            self.s_a_0 = copy.deepcopy(self.root.Controller.get_actuators_state())
            self.s_e_0 = copy.deepcopy(self.root.Controller.get_effectors_state())
            self.effector_pos_0 = []
            for effector in self.root.Controller.list_effectors_MO:
                self.effector_pos_0 += copy.deepcopy(effector.rest_position.value)[0].tolist()

        if self.monitoring_time:
            start_time = time.time()

        if self.step < self.waiting_time:
            print("[INFO]  >>  Waiting time {}/{}".format(self.step, self.waiting_time))
        else:
            if self.step == self.waiting_time:
                self.W_0 = copy.deepcopy(self.root.Controller.get_compliance_matrice_in_constraint_space())
                self.dfree_0 = copy.deepcopy(self.root.Controller.get_dfree())
                self.init_actuation()

            data_matrices = self.recover_matrices()


        if self.step >= self.waiting_time:

            data_QP_matrices = self.build_QP_matrices(data_matrices)
            actuation = self.solve_QP_problem(data_QP_matrices)
            self.apply_actuation(actuation)

            if self.monitoring_time:
                end_time = time.time()

            # Update goal position if managing trajectory
            if self.simu_results:
                eq_counter = self.step % self.waiting_eq_dt
                if eq_counter == 0:
                    self.reinit_correction()
                    self.update_goal_position()

                    # TODO: get curves and plot
        self.step+=1
        if self.monitoring_time and self.step >= self.waiting_time+1:
            self.real_time_computation+= end_time - start_time
            print("[INFO] >> Mean time after ", self.step-self.waiting_time, " steps:", self.real_time_computation/(self.step-self.waiting_time))

    def update_goal_position(self):
        if not self.config.useRigidGoal:
            self.root.Goal.GoalMO.position.value = [self.intermediate_goals[min(self.step // self.waiting_eq_dt - 1,
                                                                                len(self.intermediate_goals) - 1)].tolist()]
        else:
            pos = self.intermediate_goals[min(self.step // self.waiting_eq_dt - 1, len(self.intermediate_goals) - 1)][
                  :3].tolist()
            ang = self.intermediate_goals[min(self.step // self.waiting_eq_dt - 1, len(self.intermediate_goals) - 1)][
                  3:].tolist()
            ang = euler_to_quaternion(ang[0], ang[1], ang[2])
            self.root.Goal.GoalMO.position.value = [pos + ang]
    def update_save_data(self):
        print("[ERROR] >> You have to implement update_save_data.")
        exit(1)
    def onAnimateEndEvent(self, event):

        # Manage trajectory
        if self.simu_results:
            eq_counter = self.step % self.waiting_eq_dt
            if eq_counter == 0 and self.step // self.waiting_eq_dt - 1 <= len(self.intermediate_goals) - 1 :
                print("[WARNING]  >> Registering trajectory results.")
                self.update_save_data()

                path = "./Results/Trajectories/Data/trajectory_" + self.config.model_name + "_learned.txt" if self.type_use == "learned" else "./Results/Trajectories/Data/trajectory_" + self.config.model_name + "_computed.txt"
                with open(path, 'w') as fp:
                    json.dump(self.save_data, fp)
