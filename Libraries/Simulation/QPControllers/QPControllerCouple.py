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
        self.solver = init_QP(2*self.n_act_constraint)

    def recover_matrices(self):
        s_a_t = copy.deepcopy(self.root.Controller.get_actuators_state())
        s_e_t = copy.deepcopy(self.root.Controller.get_effectors_state())

        W_t = copy.deepcopy(self.root.Controller.get_compliance_matrice_in_constraint_space())
        dfree_t = copy.deepcopy(self.root.Controller.get_dfree())

        data_matrices = {}
        if self.type_use == "learned":
            data_matrices["Waa_1"], data_matrices["Waa_2"], data_matrices["Wea_1"], data_matrices["Wea_2"], _, _, data_matrices["dfree_a"], data_matrices["dfree_e"] = compute_QP_matrices_from_MLP_coupled_fingers(self.model, self.n_constraint, self.data_scaling, self.n_act_constraint, s_a_t, s_e_t, self.W_0, self.dfree_0, normalization_method = self.normalization_method)

            # We don't train the Finger with different value of forces applied on the effector.
            # The learned value of Wee is not revelant
            _, _, _, _,  data_matrices["Wee_1"], data_matrices["Wee_2"], _, _, _, _ = compute_QP_matrices_from_simulation_coupled_fingers(self.n_constraint,
                                                                                           self.n_act_constraint, W_t,
                                                                                           dfree_t)
        else:
            data_matrices["Waa_1"], data_matrices["Waa_2"], data_matrices["Wea_1"], data_matrices["Wea_2"], data_matrices["Wee_1"], data_matrices["Wee_2"], _, _, _, _ = compute_QP_matrices_from_simulation_coupled_fingers(self.n_constraint, self.n_act_constraint, W_t, dfree_t)

        data_matrices["s_a_t"] = s_a_t
        return data_matrices

    def build_QP_matrices(self, data_matrices):
        data_QP_matrices ={}

        P1 = self.root.Simulation.Finger1.Effectors.MechanicalObject.position.value[0] #position of effector of Finger 1
        P2 = self.root.Simulation.Finger2.Effectors.MechanicalObject.position.value[0] #position of effector of Finger 1
        Pg = self.root.Goal.GoalMO.position.value[0]  #position of the goal
        scale = np.array([0, 40, 0]) #desired space between effector

        inv_Wee = np.linalg.inv(data_matrices["Wee_1"] + data_matrices["Wee_2"])
        Wee_g = np.matmul(data_matrices["Wee_1"], inv_Wee).astype('double')

        A_1 = data_matrices["Wea_1"] - np.dot(Wee_g, data_matrices["Wea_1"])
        A_2 = np.dot(Wee_g, data_matrices["Wea_2"])
        P = P1 - Pg - np.dot(Wee_g, scale + P1 - P2)

        M = np.zeros((2 * A_1.shape[0], A_1.shape[1] + A_2.shape[1]))
        M[:A_1.shape[0], :A_1.shape[1]] = A_1.copy()
        M[A_1.shape[0]:, :A_1.shape[1]] = A_1.copy()
        M[:A_1.shape[0], A_1.shape[1]:] = A_2.copy()
        M[A_1.shape[0]:, A_1.shape[1]:] = A_2.copy()

        data_QP_matrices["H"] = np.dot(np.transpose(M), M).astype('double')

        g_1 = 2 * np.dot(np.transpose(A_1), P)
        g_2 = 2 * np.dot(np.transpose(A_2), P)
        data_QP_matrices["g"] = np.concatenate([g_1, g_2])

        a = np.zeros((2 * self.n_act_constraint, 2 * self.n_act_constraint))
        a[:self.n_act_constraint, :self.n_act_constraint] = data_matrices["Waa_1"].copy()
        a[self.n_act_constraint:, self.n_act_constraint:] = data_matrices["Waa_2"].copy()
        data_QP_matrices["A"] = a.astype('double')

        data_QP_matrices["lb"] = np.array([0 for i in range(2 * self.n_act_constraint)]).astype('double')
        data_QP_matrices["ub"] = np.array([np.inf for i in range(2 * self.n_act_constraint)]).astype(
            'double')  # Upper bound for actuation effort constraint - specific to a cable

        data_QP_matrices["lbA"] = (np.array([-np.inf for i in range(2 * self.n_act_constraint)])).astype('double')
        data_QP_matrices["ubA"] = (np.array([np.inf for i in range(2 * self.n_act_constraint)])).astype('double')


        #To compute lambda_e
        data_QP_matrices["inv_Wee"] = inv_Wee
        data_QP_matrices["Wea_1"] = data_matrices["Wea_1"]
        data_QP_matrices["Wea_2"] = data_matrices["Wea_2"]
        data_QP_matrices["scale"] = scale
        data_QP_matrices["P1"] = P1
        data_QP_matrices["P2"] = P2

        return data_QP_matrices

    def solve_QP_problem(self, data_QP_matrices):

        self.d_lambda_a = solve_QP(self.solver, data_QP_matrices["H"], data_QP_matrices["g"], data_QP_matrices["A"],
                                   data_QP_matrices["lb"], data_QP_matrices["ub"], data_QP_matrices["lbA"],
                                   data_QP_matrices["ubA"], is_init=(self.step != self.waiting_time))

        Wea1Wea2 = np.zeros((data_QP_matrices["Wea_1"].shape[0], data_QP_matrices["Wea_1"].shape[1] + data_QP_matrices["Wea_2"].shape[1]))
        Wea1Wea2[:, :data_QP_matrices["Wea_1"].shape[1]] = data_QP_matrices["Wea_1"].copy()
        Wea1Wea2[:, data_QP_matrices["Wea_1"].shape[1]:] = -data_QP_matrices["Wea_2"].copy()
        Weedelta_e = data_QP_matrices["scale"] + data_QP_matrices["P1"] - data_QP_matrices["P2"] + np.dot(Wea1Wea2, np.transpose(self.d_lambda_a))
        self.d_lambda_e = -np.dot(data_QP_matrices["inv_Wee"], Weedelta_e)

        # Smooth actuation update
        factor = 0.5
        self.lambda_a = self.lambda_a + factor * (self.d_lambda_a - self.lambda_a)
        self.lambda_e = self.lambda_e + factor * (self.d_lambda_e - self.lambda_e)

        return {}

    def init_actuation(self):
        n_fingers = 2
        self.n_constraint = int(len(self.W_0) / n_fingers)
        self.lambda_a, self.lambda_e = np.zeros(2 * self.n_act_constraint), np.zeros(self.n_constraint - self.n_act_constraint)

    def apply_actuation(self, data_actuation):
        self.root.Simulation.Finger1.cables.cable1.CableConstraint.value.value = [self.lambda_a[0]]
        self.root.Simulation.Finger1.cables.cable2.CableConstraint.value.value = [self.lambda_a[1]]
        self.root.Simulation.Finger2.cables.cable1.CableConstraint.value.value = [self.lambda_a[2]]
        self.root.Simulation.Finger2.cables.cable2.CableConstraint.value.value = [self.lambda_a[3]]

        self.root.Simulation.Finger1.Effectors.ConstraintPoint.imposedValue.value = self.lambda_e
        self.root.Simulation.Finger2.Effectors.ConstraintPoint.imposedValue.value = -self.lambda_e

        # Update the lambda vector before applying corrective motion in the scene
        # For now, we register our result on top of the one computed by SOFA QPSolver implementation
        lambda_constraint_vector = self.root.Controller.constraint_solver.lambda_force()

        # Order in scene should be [ActatorsFinger1, EffectorsFinger1, ActuatorsFinger2, EffectorsFinger3]
        for i in range(2):
            lambda_constraint_vector[i] = self.lambda_a[i]
        for i in range(3):
            lambda_constraint_vector[2 + i] = self.lambda_e[i]
        for i in range(2):
            lambda_constraint_vector[5 + i] = self.lambda_a[2 + i]
        for i in range(3):
            lambda_constraint_vector[7 + i] = -self.lambda_e[i]

        self.root.Controller.constraint_solver_setter.set_lambda_force(lambda_constraint_vector)
    def onAnimateEndEvent(self, event):
        pass



###############################################################################
############### Functions for the 2 coupled Fingers cases #####################
###############################################################################
def compute_QP_matrices_from_MLP_coupled_fingers(MLP_model, n_constraint, scaling, n_act_constraint, s_a_t, s_e_t, W_t, dfree_t, normalization_method):
    """
    Rebuild QP matrices using the MLP model for the coupled Fingers case

    W = |Waa  Wae|
        |Wea  Wee|

    Parameters
    ----------
    MLP_model: MLP
        Learned encode, process and decode components
    n_constraint: int
        Total number of constraints
    scaling = [min_features_X, max_features_X, min_features_Y, max_features_Y]: list of list of numpy arrays
        Min/Max scaling for each component for each matrice
    n_act_constraint: int
        Number of constraints for actuators
    s_a_t: list of float
        Actuation displacement state
    W_t: numpy array
        Compliance matrice projected in constraint space without constraint
    dfree_t: numpy array
        Displacement in free configuration without any actuation
    normalization_method: str
        Method used for normalizing data:
            - None: No normalization method used
            - MinMax: Use minimum and maximum value for normalizing each feature
            - Std: Use mean and standard deviation for normalizing each feature

    Outputs
    ----------
    Waa_1: numpy array
        Compliance matrice projected in actuator/actuator constraint space for Finger 1
    Waa_2: numpy array
        Compliance matrice projected in actuator/actuator constraint space for Finger 2
    Wea_1: numpy array
        Compliance matrice projected in effector/actuators constraint space for Finger 1
    Wea_2: numpy array
        Compliance matrice projected in effector/actuators constraint space for Finger 2
    Wee_1: numpy array
        Compliance matrice projected in effector/effector constraint space for Finger 1
    Wee_2: numpy array
        Compliance matrice projected in effector/effector constraint space for Finger 2
    dfree_a: numpy array
        Predicted actuators displacement without any actuation
    dfree_e: numpy array
        Predicted effectors displacement without any actuation
    """
    n_effectors = n_constraint - n_act_constraint

    W_0 = W_t[:n_constraint, :n_constraint]
    dfree_0 = dfree_t[:n_constraint]

    # Prediction
    MLP_lib = importlib.import_module("Libraries.Learning.MLP.learning_tools")
    X1 = [torch.tensor(W_0), torch.tensor(dfree_0), torch.tensor(s_a_t[:n_act_constraint] + s_e_t[:n_effectors])]
    X2 = [torch.tensor(W_0), torch.tensor(dfree_0), torch.tensor(s_a_t[n_act_constraint:] + s_e_t[n_effectors:])]
    Y = [torch.tensor(W_0), torch.tensor(dfree_0)]

    # Rescale data before providing it to the NN
    if normalization_method == "Std":
        X1, _ = MLP_lib.create_data_std(X1, Y, scaling[0], scaling[1], scaling[2], scaling[3])
        X2, _ = MLP_lib.create_data_std(X2, Y, scaling[0], scaling[1], scaling[2], scaling[3])
    elif normalization_method == "MinMax":
        X1, _ = MLP_lib.create_data_minmax(X1, Y, scaling[0], scaling[1], scaling[2], scaling[3])
        X2, _ = MLP_lib.create_data_minmax(X2, Y, scaling[0], scaling[1], scaling[2], scaling[3])
    else:
        X1, _ = MLP_lib.create_data(X1, Y)
        X2, _ = MLP_lib.create_data(X2, Y)

    Y1= MLP_model(X1)
    Y2= MLP_model(X2)

    ### Compute d_free from prediction ###
    dfree = Y1[-n_constraint:].detach().numpy()
    if normalization_method == "Std":
        dfree = dfree * (scaling[3][1]) + scaling[2][1] # Rescale dfree
    elif normalization_method == "MinMax":
        dfree = dfree * (scaling[3][1]  - scaling[2][1]) + scaling[2][1] # Rescale dfree

    dfree_a = dfree[0:n_act_constraint]
    dfree_e = dfree[n_act_constraint:]

    ### Compute W from prediction ###
    W1_pred = Y1[:-n_constraint].detach().numpy()
    W1 = np.zeros((n_constraint, n_constraint))
    W1[np.triu_indices(n=n_constraint)] = W1_pred
    W1[np.tril_indices(n=n_constraint, k=-1)] = W1.T[np.tril_indices(n=n_constraint, k=-1)]
    W1 = W1.reshape(-1)
    if normalization_method == "Std":
        W1 = W1 * (scaling[3][0].reshape(-1)) + scaling[2][0].reshape(-1) # Rescale W
    elif normalization_method == "MinMax":
        W1 = W1 * (scaling[3][0].reshape(-1) - scaling[2][0].reshape(-1)) + scaling[2][0].reshape(-1) # Rescale W
    W1 = W1.reshape(n_constraint,n_constraint)
    Waa_1 = W1[0:n_act_constraint, 0:n_act_constraint]
    Wea_1 = W1[n_act_constraint:, 0:n_act_constraint]
    Wee_1 = W1[n_act_constraint:, n_act_constraint:]

    W2_pred = Y2[:-n_constraint].detach().numpy()
    W2 = np.zeros((n_constraint, n_constraint))
    W2[np.triu_indices(n=n_constraint)] = W2_pred
    W2[np.tril_indices(n=n_constraint, k=-1)] = W2.T[np.tril_indices(n=n_constraint, k=-1)]
    W2 = W2.reshape(-1)
    if normalization_method == "Std":
        W2 = W2 * (scaling[3][0].reshape(-1)) + scaling[2][0].reshape(-1) # Rescale W
    elif normalization_method == "MinMax":
        W2 = W2 * (scaling[3][0].reshape(-1) - scaling[2][0].reshape(-1)) + scaling[2][0].reshape(-1) # Rescale W
    W2 = W2.reshape(n_constraint,n_constraint)
    Waa_2 = W2[0:n_act_constraint, 0:n_act_constraint]
    Wea_2 = W2[n_act_constraint:, 0:n_act_constraint]
    Wee_2 = W2[n_act_constraint:, n_act_constraint:]

    # Compute second Finger matrices from first Finger matrices
    # The second matrice is rotated
    new_Waa, new_Wea, new_Wee = Waa_2.copy(), Wea_2.copy(), Wee_2.copy()
    for i, j in zip([0, 0, 1, 1], [0, 1, 0, 1]):
        new_Wea[i][j] = -new_Wea[i][j]
    for i, j in zip([0, 1, 2, 2, 0, 1], [2, 2, 0, 1, 1,0]):
        new_Wee[i][j] = -new_Wee[i][j]

    Waa_2, Wea_2, Wee_2 = new_Waa, new_Wea, new_Wee

    return Waa_1, Waa_2, Wea_1, Wea_2, Wee_1, Wee_2, dfree_a, dfree_e


def compute_QP_matrices_from_simulation_coupled_fingers(n_constraint, n_act_constraint, W_t, dfree_t):
    """
    Rebuild QP matrices for QP solving

    W = |Waa  Wae|
        |Wea  Wee|

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
    Waa_1: numpy array
        Compliance matrice projected in actuator/actuator constraint space for Finger 1
    Waa_2: numpy array
        Compliance matrice projected in actuator/actuator constraint space for Finger 2
    Wea_1: numpy array
        Compliance matrice projected in effector/actuators constraint space for Finger 1
    Wea_2: numpy array
        Compliance matrice projected in effector/actuators constraint space for Finger 2
    Wee_1: numpy array
        Compliance matrice projected in effector/effector constraint space for Finger 1
    Wee_2: numpy array
        Compliance matrice projected in effector/effector constraint space for Finger 2
    dfree_a_1: numpy array
        Predicted actuators displacement without any actuation for Finger 1
    dfree_a_2: numpy array
        Predicted actuators displacement without any actuation for Finger 2
    dfree_e_1: numpy array
        Predicted effectors displacement without any actuation for Finger 1
    dfree_e_2: numpy array
        Predicted effectors displacement without any actuation for Finger 2
    """

    W_1 = W_t[:n_constraint, :n_constraint]
    W_2 = W_t[n_constraint:, n_constraint:]

    ### Compute d_free_i  ###
    dfree_1 = dfree_t[:n_constraint]
    dfree_2 = dfree_t[n_constraint:]

    dfree_a_1 = dfree_1[0:n_act_constraint]
    dfree_e_1 = dfree_1[n_act_constraint:]
    dfree_a_2 = dfree_2[0:n_act_constraint]
    dfree_e_2 = dfree_2[n_act_constraint:]

    ### Compute W_ij ###
    Waa_1 = W_1[0:n_act_constraint, 0:n_act_constraint]
    Waa_2 = W_2[0:n_act_constraint, 0:n_act_constraint]

    Wea_1 = W_1[n_act_constraint:, :n_act_constraint]
    Wea_2 = W_2[n_act_constraint:, :n_act_constraint]

    Wee_1 = W_1[n_act_constraint:, n_act_constraint:]
    Wee_2 = W_2[n_act_constraint:, n_act_constraint:]

    return Waa_1, Waa_2, Wea_1, Wea_2, Wee_1, Wee_2, dfree_a_1, dfree_e_1, dfree_a_2, dfree_e_2
