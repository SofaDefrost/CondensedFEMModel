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

from BaseQPController import BaseQPController
from database import query_simulation_data, query_sampling_stats, query_recover_W_dfree_from_s_a
from utils import ask_user_sampling_stat_input
from QPproblem import init_QP, solve_QP


class QPController(BaseQPController):
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
        BaseQPController.__init__(self, *args, **kwargs)


        if not(self.config.scene_config["is_force"]):
            print("[ERROR]  >> Set is_force in the config.")
            exit(1)


    def init_QP(self):
        self.solver = init_QP(self.n_act_constraint)
    def recover_matrices(self):
        if not self.config.useRigidGoal:
            self.config.scene_config["goalPos"] = self.root.Goal.GoalMO.position.value[:].tolist()
        else:
            pos_0 = self.root.Goal.Goal_0.GoalMO.position.value[:].tolist()
            pos_1 = self.root.Goal.Goal_1.GoalMO.position.value[:].tolist()
            pos_2 = self.root.Goal.Goal_2.GoalMO.position.value[:].tolist()
            self.config.scene_config["goalPos"] = pos_0 + pos_1 + pos_2

        s_a_t = copy.deepcopy(self.root.Controller.get_actuators_state())
        s_e_t = copy.deepcopy(self.root.Controller.get_effectors_state())

        data_matrices = {}
        if self.type_use == "learned":
            data_matrices["Waa"], data_matrices["Wea"], data_matrices["dfree_a"], data_matrices["dfree_e"] = compute_QP_matrices_from_learning(self.network_name, self.model,
                                                                           self.n_constraint, self.data_scaling,
                                                                           self.n_act_constraint, s_a_t, s_e_t,
                                                                           self.W_0, self.dfree_0, self.effector_pos_0,
                                                                           self.config.scene_config["goalPos"],
                                                                           normalization_method=self.normalization_method)
        else:
            W_t = copy.deepcopy(self.root.Controller.get_compliance_matrice_in_constraint_space())
            dfree_t = copy.deepcopy(self.root.Controller.get_dfree())
            data_matrices["Waa"], data_matrices["Wea"], data_matrices["dfree_a"], data_matrices["dfree_e"] = compute_QP_matrices_from_simulation(self.n_act_constraint, W_t, dfree_t)

        data_matrices["dfree_e"] = self.apply_correction(dfree=data_matrices["dfree_e"], Wea=data_matrices["Wea"])
        data_matrices["s_a_t"] = s_a_t
        return data_matrices

    def build_QP_matrices(self, data_matrices):
        use_epsilon = True
        delta_a_var = 1
        data_QP_matrices = {}

        # Init QP problem with our data at dt
        if use_epsilon:
            epsilon = 0.01 * np.linalg.norm(np.dot(np.transpose(data_matrices["Waa"]), data_matrices["Waa"]), ord=1) / np.linalg.norm(data_matrices["Waa"], ord=1)
        else:
            epsilon = 0.0

        data_QP_matrices["H"] = np.dot(np.transpose(data_matrices["Wea"]), data_matrices["Wea"]).astype('double') + epsilon * (data_matrices["Waa"]).astype('double')
        data_QP_matrices["g"] = np.dot(np.transpose(data_matrices["Wea"]), data_matrices["dfree_e"]).astype('double')
        data_QP_matrices["A"] = data_matrices["Waa"].astype('double')

        data_QP_matrices["lb"] = np.array([0 for i in range(len(data_matrices["dfree_a"]))]).astype(
            'double')  # Lower bound for actuation effort constraint - specific to a cable
        data_QP_matrices["ub"] = np.array([np.inf for i in range(len(data_matrices["dfree_a"]))]).astype(
            'double')  # Upper bound for actuation effort constraint - specific to a cable

        # Diamond:
        # data_QP_matrices["lbA"] = (np.array([-np.inf for i in range(len(data_matrices["dfree_a"]))])).astype('double')
        # data_QP_matrices["ubA"] = (np.array([20 for i in range(len(data_matrices["dfree_a"]))])).astype('double')

        # Finger
        data_QP_matrices["lbA"] = (np.array([-np.inf for i in range(len(data_matrices["dfree_a"]))])).astype('double')
        data_QP_matrices["ubA"] = (np.array([np.inf for i in range(len(data_matrices["dfree_a"]))])).astype('double')

        # Trunk
        # data_QP_matrices["lbA"] = (np.array([ max(self.min_s_a[i], abs(data_matrices["s_a_t"][i] - self.s_a_0[i]) - delta_a_var) for i in range(len(data_matrices["dfree_a"]))]) - data_matrices["dfree_a"]).astype('double') # Lower bound for actuator displacement constraint - specific to a cable
        # data_QP_matrices["ubA"] = (np.array([ min(self.max_s_a[i], abs(data_matrices["s_a_t"][i] - self.s_a_0[i]) + delta_a_var) for i in range(len(data_matrices["dfree_a"]))]) - data_matrices["dfree_a"]).astype('double') # Upper bound for actuator displacement constraint - specific to a cable

        return data_QP_matrices

    def solve_QP_problem(self, data_QP_matrices):
        data_actuation = {}

        data_actuation["lambda_a"] =solve_QP(self.solver, data_QP_matrices["H"], data_QP_matrices["g"],
                                            data_QP_matrices["A"], data_QP_matrices["lb"], data_QP_matrices["ub"],
                                            data_QP_matrices["lbA"], data_QP_matrices["ubA"], is_init=(self.step != self.waiting_time))

        return data_actuation

    def init_actuation(self):
        self.lambda_a = np.zeros(self.n_act_constraint)
    def apply_actuation(self, data_actuation):
        self.root.Controller.apply_actions(data_actuation["lambda_a"])

        # Update the lambda vector before applying corrective motion in the scene
        # For now, we register our result on top of the one computed by SOFA QPSolver
        lambda_constraint_vector = copy.deepcopy(self.root.Controller.constraint_solver.lambda_force())
        for i in range(len(data_actuation["lambda_a"])):
            lambda_constraint_vector[i] = data_actuation["lambda_a"][i]  # Replace constraint vector values with our computed values
        self.root.Controller.constraint_solver_setter.set_lambda_force(lambda_constraint_vector)



    def update_save_data(self):
        goal_pos = self.root.Goal.GoalMO.position.value.tolist()[0]
        effector_pos = self.root.Controller.get_effectors_positions()[0].tolist()[0]

        self.save_data["pos_effector"]["x"].append(effector_pos[0])
        self.save_data["pos_effector"]["y"].append(effector_pos[1])
        self.save_data["pos_effector"]["z"].append(effector_pos[2])

        self.save_data["pos_goal"]["x"].append(goal_pos[0])
        self.save_data["pos_goal"]["y"].append(goal_pos[1])
        self.save_data["pos_goal"]["z"].append(goal_pos[2])




###############################################################################
####################### Functions to recover matrices #########################
###############################################################################

### From simulation
def compute_QP_matrices_from_simulation(n_act_constraint, W_t, dfree_t):
    """
    Rebuild QP matrices for QP solving

    W = |Waa  Wae|
        |Wea  Wee|

    Parameters
    ----------
    n_act_constraint: int
        Number of constraints for actuators
    W_t: numpy array
        Compliance matrice projected in constraint space
    dfree_t: numpy array
        Displacement in free configuration

    Outputs
    ----------
    Waa: numpy array
        Compliance matrice projected in actuator/actuator constraint space
    Wea: numpy array
        Compliance matrice projected in effector/actuators constraint space
    dfree_a: numpy array
        Actuators displacement
    dfree_e: numpy array
        Effectors displacement
    """

    ### Compute d_free_i  ###
    dfree_a = dfree_t[0:n_act_constraint]
    dfree_e = dfree_t[n_act_constraint:]

    ### Compute W_ij ###
    Waa = W_t[0:n_act_constraint, 0:n_act_constraint]
    Wea = W_t[n_act_constraint:, 0:n_act_constraint]

    return Waa, Wea, dfree_a, dfree_e


### From learning
def compute_QP_matrices_from_learning(network_name, model, n_constraint, scaling, n_act_constraint, s_a_t, s_e_t, W_t, dfree_t, effector_pos_0, goals_pos, normalization_method = "MinMax"):
    """
    Rebuild QP matrices using the neural network model

    W = |Waa  Wae|
        |Wea  Wee|

    Parameters
    ----------
    network_name: str
        The name of the neural network we use in [MLP]
    model: neural network
        The neural network we used to predict compliance and dfree value.
    n_constraint: int
        Total number of constraints
    scaling = list of list of numpy arrays
        Scaling for each component for each matrice
    n_act_constraint: int
        Number of constraints for actuators
    s_a_t: list of float
        Actuation displacement state
    s_e_t: list of float
        Effector displacement state
    W_t: numpy array
        Compliance matrice projected in constraint space without constraint
    dfree_t: numpy array
        Displacement in free configuration without any actuation
    effector_pos_0: list of numpy array
        Rest positions of the effectors
    goals_pos: list of numpy array
        Positions of the goals
    normalization_method: str
        Method used for normalizing data:
            - None: No normalization method used
            - MinMax: Use minimum and maximum value for normalizing each feature
            - Std: Use mean and standard deviation for normalizing each feature


    Outputs
    ----------
    Waa: numpy array
        Predicted compliance matrice projected in actuator/actuator constraint space
    Wea: numpy array
        Predicted compliance matrice projected in effector/actuators constraint space
    dfree_a: numpy array
        Predicted actuators displacement without any actuation
    dfree_e: numpy array
        Predicted effectors displacement without any actuation
    """

    n_effectors = n_constraint - n_act_constraint # May change when considering collision

    if network_name == "MLP" or network_name == "doubleMLP":
        MLP_lib = importlib.import_module("Libraries.Learning.MLP.learning_tools")
        X = [torch.tensor(W_t), torch.tensor(dfree_t), torch.tensor(s_a_t + s_e_t)]
        Y = [torch.tensor(W_t), torch.tensor(dfree_t)]

        # Rescale data before providing it to the NN
        if normalization_method == "Std":
            X, _ = MLP_lib.create_data_std(X, Y, scaling[0], scaling[1], scaling[2], scaling[3])
        elif normalization_method == "MinMax":
            X, _ = MLP_lib.create_data_minmax(X, Y, scaling[0], scaling[1], scaling[2], scaling[3])
        else:
            X, _ = MLP_lib.create_data(X, Y)

        # Prediction
        Y = model(X)

        dfree = Y[-n_constraint:].detach().numpy()
        W_pred = Y[:-n_constraint].detach().numpy()

        W = np.zeros((n_constraint, n_constraint))
        W[np.triu_indices(n=n_constraint)] = W_pred
        W[np.tril_indices(n=n_constraint, k=-1)] = W.T[np.tril_indices(n=n_constraint, k=-1)]
        W = W.reshape(-1)

    ### Compute d_free from prediction ###
    if normalization_method == "Std":
        dfree = dfree * (scaling[3][1]) + scaling[2][1] # Rescale dfree
    elif normalization_method == "MinMax":
        dfree = dfree * (scaling[3][1]  - scaling[2][1]) + scaling[2][1] # Rescale dfree

    dfree_a = dfree[0:n_act_constraint]
    dfree_e = dfree[n_act_constraint:]

    flat_goals_pos = [coord for pos in goals_pos for coord in pos]
    for i in range(len(dfree_e)):
        dfree_e[i] += effector_pos_0[i] - flat_goals_pos[i] # (Init_pos + Relative_Disp) - Goal_pos

    ### Compute W from prediction ###
    if normalization_method == "Std":
        W = W * (scaling[3][0].reshape(-1)) + scaling[2][0].reshape(-1) # Rescale W
    elif normalization_method == "MinMax":
        W = W * (scaling[3][0].reshape(-1) - scaling[2][0].reshape(-1)) + scaling[2][0].reshape(-1) # Rescale W
    W = W.reshape(n_constraint,n_constraint)

    Waa = W[0:n_act_constraint, 0:n_act_constraint]
    Wea = W[n_act_constraint:, 0:n_act_constraint]

    return Waa, Wea, dfree_a, dfree_e
