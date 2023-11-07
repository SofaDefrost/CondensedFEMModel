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

from BaseQPController import BaseQPController
from QPproblem import init_QP, solve_QP
from database import query_simulation_data, query_sampling_stats, query_recover_W_dfree_from_s_a
from utils import ask_user_sampling_stat_input, create_bloc
from geometry import euler_to_quaternion, quaternion_to_euler

class QPController(BaseQPController):
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
        BaseQPController.__init__(self, *args, **kwargs)


        # Define contact points and actuators
        list_actuators = self.root.Controller.get_actuators()
        self.list_contact_finger = self.root.Controller.list_contacts
        self.list_contact_cube = self.root.Controller.list_cube_contacts
        self.list_actuator = self.root.Controller.list_actuators

        self.n_constraint = len(self.config.get_actuators_variables())  + len(self.config.get_contacts_variables()) # Number of constraints for one finger
        self.n_act_constraint = int(len(self.list_actuator) / self.config.nb_robot) # Number of actuators for one finger

        self.save_data_finger_effectors = {"pos_effector": {"x": [], "y": [], "z": []}}

        self.list_Wcc = [None for _ in range(self.config.nb_robot)]
        self.list_Wca = [None for _ in range(self.config.nb_robot)]
        self.list_Waa = [None for _ in range(self.config.nb_robot)]
        self.list_delta_a_free = [None for _ in range(self.config.nb_robot)]

    def init_QP(self):
        self.solver = init_QP(self.config.nb_robot*self.n_act_constraint)
    def recover_matrices(self):
        s_a_t = copy.deepcopy(self.root.Controller.get_actuators_state())
        s_e_t = copy.deepcopy(self.root.Controller.get_effectors_state())

        data_matrices = {}
        if self.type_use == "learned":
            self.list_Waa, self.list_Wca, self.list_Wcc, self.list_delta_a_free, self.list_delta_c_free = compute_mechanical_matrices_from_MLP(
                self.model, self.n_constraint, self.data_scaling, self.n_act_constraint, s_a_t, self.W_0, self.dfree_0,
                self.config.nb_robot,
                normalization_method=self.normalization_method)
        else:
            W_t = copy.deepcopy(self.root.Controller.get_compliance_matrice_in_constraint_space())
            dfree_t = copy.deepcopy(self.root.Controller.get_dfree())
            self.list_Wcc, self.list_Wca, self.list_Waa, self.list_delta_a_free, self.list_delta_c_free = self.recover_from_simulation(self.config.nb_robot, self.n_constraint, self.n_act_constraint, W_t, dfree_t)


        data_matrices["X_cube"] = self.get_cube_pos()
        self.X_goal = np.array(self.config.scene_config["goalPos"])
        xc_finger = self.get_finger_contact_pos_vector()
        data_matrices["xc_cube"] = self.get_cube_contact_pos_vector()

        Cik = self.compute_Cik().tolist()
        data_matrices["Jc"] = self.compute_Jc(Cik)
        data_matrices["Wcc"], data_matrices["Wca"], data_matrices["Waa"], data_matrices["delta_free_a"] = self.compute_matrices()

        D = self.compute_D(data_matrices["Jc"], data_matrices["Wcc"])
        A = self.compute_A(data_matrices["Jc"], data_matrices["Wcc"], data_matrices["Wca"], D)

        data_matrices["dfree_c"] = np.concatenate(self.list_delta_c_free, axis=0)
        data_matrices["dfree_a"] = np.concatenate(self.list_delta_a_free, axis=0)
        B = self.compute_B_from_dfree_c(data_matrices["X_cube"], self.lambda_c, D, data_matrices["Jc"], data_matrices["Wcc"], data_matrices["dfree_c"], data_matrices["xc_cube"])
        b = self.compute_b(B, self.X_goal)

        MASK_ROTATION = True
        if MASK_ROTATION:
            # list_dirs = [0.01, 0.01, 0.01, 1, 1, 1]
            if self.config.nb_robot == 3:
                list_dirs = [1, 1, 1, 0, 0, 0]
            else:
                list_dirs = [0, 1, 0, 0, 0, 0]
            mask_matrix = self.compute_mask_matrice(list_dirs)
            data_matrices["A"] = np.matmul(mask_matrix, A)
            data_matrices["b"] = np.matmul(mask_matrix, b)

        else:
            data_matrices["A"] = A
            data_matrices["b"] = b

        #To update pos
        data_matrices["complete_A"] = A
        data_matrices["B"] = B

        return data_matrices

    def recover_from_simulation(self, nb_robot, n_constraint, n_act_constraint, W_t, dfree_t):
        return compute_mechanical_matrices_from_simulation(nb_robot, n_constraint, n_act_constraint, W_t, dfree_t)

    def build_QP_matrices(self, data_matrices):
        data_QP_matrices = {}

        use_epsilon = False
        if use_epsilon:
            epsilon = 0.01 * np.linalg.norm(np.dot(np.transpose(data_matrices["Wca"]), data_matrices["Wca"]), ord=1) / np.linalg.norm(data_matrices["Waa"], ord=1)
        else:
            epsilon = 0.0

        data_QP_matrices["H"] = 2*np.matmul(data_matrices["A"].T, data_matrices["A"]) + epsilon *(data_matrices["Waa"]).astype('double')
        data_QP_matrices["g"] = 2*np.matmul(data_matrices["A"].T, data_matrices["b"])
        data_QP_matrices["C"] = data_matrices["Waa"]

        ### Actuation constraints
        data_QP_matrices["l"] = np.zeros(data_matrices["delta_free_a"].shape) + 0.0 # Actuation should be more than 0 - We can't push with a cable
        data_QP_matrices["u"] = np.zeros(data_matrices["delta_free_a"].shape) + np.infty
        #u = np.zeros(data_matrices["delta_free_a"].shape) + 10000.0 # Constraint maximum actuation

        ### Actuation displacement constraints
        data_QP_matrices["lb"] = (np.array([-np.inf for i in range(len(data_matrices["delta_free_a"]))])).astype('double')
        data_QP_matrices["ub"] = (np.array([np.inf for i in range(len(data_matrices["delta_free_a"]))])).astype('double')

        # To update pos
        data_QP_matrices["A"] = data_matrices["complete_A"]
        data_QP_matrices["B"] = data_matrices["B"]
        data_QP_matrices["X_cube"] = data_matrices["X_cube"]
        data_QP_matrices["xc_cube"]=data_matrices["xc_cube"]
        data_QP_matrices["Jc"]= data_matrices["Jc"]
        data_QP_matrices["Wcc"]=data_matrices["Wcc"]
        data_QP_matrices["Wca"]=data_matrices["Wca"]
        data_QP_matrices["Waa"]=data_matrices["Waa"]
        data_QP_matrices["delta_free_a"]=data_matrices["delta_free_a"]

        data_QP_matrices["dfree_c"]=data_matrices["dfree_c"]
        data_QP_matrices["dfree_a"]=data_matrices["dfree_a"]

        return data_QP_matrices


    def solve_QP_problem(self, data_QP_matrices):
        new_lambda_a = solve_QP(self.solver, data_QP_matrices["H"], data_QP_matrices["g"], data_QP_matrices["C"],
                                data_QP_matrices["l"], data_QP_matrices["u"], data_QP_matrices["lb"], data_QP_matrices["ub"],
                                is_init=(self.step != self.waiting_time))


        # Compute new object position
        new_X_cube = self.compute_X(data_QP_matrices["A"], data_QP_matrices["B"], new_lambda_a)

        # Compute new contact forces
        new_lambda_c = self.compute_lambda_c_from_dfree_c(data_QP_matrices["X_cube"], new_X_cube, data_QP_matrices["Wcc"],
                                                        data_QP_matrices["Wca"], data_QP_matrices["Jc"], new_lambda_a,
                                                          data_QP_matrices["xc_cube"], data_QP_matrices["dfree_c"])

        # Update lambda_a, X_cube and lambda_c
        factor = 1
        self.lambda_a = self.lambda_a + factor * (new_lambda_a - self.lambda_a)
        self.lambda_c = self.lambda_c + factor * (new_lambda_c - self.lambda_c)

        new_X_cube = self.compute_X(data_QP_matrices["A"], data_QP_matrices["B"], self.lambda_a)  # Recompute object position with correction
        self.update_object_position(new_X_cube)

    def init_actuation(self):
        self.X_goal = np.array(self.config.scene_config["goalPos"])
        self.lambda_a = np.zeros(self.n_act_constraint * self.config.nb_robot)
        self.lambda_c = np.zeros((self.n_constraint - self.n_act_constraint) * self.config.nb_robot)
    def apply_actuation(self, data_actuation):
        assert len(self.list_actuator) == self.lambda_a.shape[0]
        for i, actuator in enumerate(self.list_actuator):
            actuator.value.value = [self.lambda_a[i]]

        for i, point_finger in enumerate(self.list_contact_finger):
            point_cube = self.list_contact_cube[i]
            point_finger.ConstraintPoint.imposedValue.value = + self.lambda_c[
                                                                   3 * i:3 * (i + 1)]  # self.lambda_c[3*i:3*(i+1)]
            point_cube.ConstraintPoint.imposedValue.value = - self.lambda_c[
                                                                 3 * i:3 * (i + 1)]  # - self.lambda_c[3*i:3*(i+1)]

        # Update the lambda vector before applying corrective motion in the scene
        # For now, we register our result on top of the one computed by SOFA QPSolver implementation
        lambda_constraint_vector = copy.deepcopy(self.root.Controller.constraint_solver.lambda_force())

        # Order in scene should be [ActatorsRobot1, ContactsRobot1, ... , ActuatorsRobotN, ContactsRobotN,
        #                           ContactCube1, ..., ContactCubeN]
        # Robots Constraints
        n_contact_constraint = self.n_constraint - self.n_act_constraint
        for robot_id in range(self.config.nb_robot):
            # Actuation
            for i in range(self.n_act_constraint):
                lambda_constraint_vector[robot_id * self.n_constraint + i] = self.lambda_a[
                    robot_id * self.n_act_constraint + i]
            # Contacts
            for i in range(n_contact_constraint):
                lambda_constraint_vector[robot_id * self.n_constraint + self.n_act_constraint + i] = + self.lambda_c[
                    robot_id * n_contact_constraint + i]

        # Object Contacts
        for i in range(self.config.nb_robot * n_contact_constraint):
            lambda_constraint_vector[self.config.nb_robot * self.n_constraint + i] = - self.lambda_c[i]

        self.root.Controller.constraint_solver_setter.set_lambda_force(lambda_constraint_vector)
    def update_save_data(self):
        goal_pos = self.config.scene_config["goalPos"]
        effector_pos = self.root.Cube.MechanicalObject.position.value[0]

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


    def get_cube_pos(self):
        # cube_pos = self.root.Cube.CoM.MechanicalObject.position.value[0]
        cube_pos = self.root.Cube.MechanicalObject.position.value[0]
        cube_trans = cube_pos[:3].tolist()
        [x, y, z, w] = cube_pos[3:].tolist()

        #Convert quaternion to euler
        [roll_x, pitch_y, yaw_z] = quaternion_to_euler(x, y, z, w)
        return np.array(cube_trans + [roll_x, pitch_y, yaw_z])

    def get_finger_contact_pos_vector(self):
        list_contact_pos = []
        for contact_point in self.list_contact_finger:
            list_contact_pos += contact_point.MechanicalObject.position.value[0, :3].tolist()
        return np.array(list_contact_pos)

    def get_cube_contact_pos_vector(self):
        list_contact_pos = []
        for contact_point in self.list_contact_cube:
            list_contact_pos += contact_point.MechanicalObject.position.value[0, :3].tolist()
        return np.array(list_contact_pos)

    def compute_b(self, B, X_goal):
        return B - X_goal

    def compute_D(self, Jc, Wcc):
        Wcc_inv = np.linalg.inv(Wcc)
        inter_1 = np.matmul(Wcc_inv, Jc)
        res = np.matmul(Jc.T, inter_1)
        return np.linalg.inv(res)

    def compute_Cik(self):
        pos = []
        for contact in self.list_contact_cube:
            pos.append(contact.MechanicalObject.position.value[0, :3].tolist())
        X_cube = self.get_cube_pos()[:3]
        return X_cube - np.array(pos)

    def compute_Jc(self, Cik):
        n_points = int(self.lambda_c.shape[0] / 3)
        Jc = []
        for i in range(n_points):
            Jc.append([1, 0, 0, 0, -Cik[i][2], Cik[i][1]])
            Jc.append([0, 1, 0, Cik[i][2], 0, -Cik[i][0]])
            Jc.append([0, 0, 1, -Cik[i][1], Cik[i][0], 0])
        return np.array(Jc)

    def compute_matrices(self):
        Wcc = create_bloc(self.list_Wcc)
        Wca = create_bloc(self.list_Wca)
        Waa = create_bloc(self.list_Waa)
        delta_free_a = np.concatenate(self.list_delta_a_free)

        return Wcc, Wca, Waa, delta_free_a

    def compute_A(self, Jc, Wcc, Wca, D):
        Wcc_inv = np.linalg.inv(Wcc)
        JcWcc_inv = np.matmul(Jc.T, Wcc_inv)
        inter_1 = np.matmul(JcWcc_inv, Wca)
        A = np.matmul(D, inter_1)
        return A

    def compute_B_from_dfree_c(self, X_cube, lambda_c, D, Jc, Wcc, dfree_c, xc_cube):

        Wcc_inv = np.linalg.inv(Wcc)
        JcWcc_inv = np.matmul(Jc.T, Wcc_inv)
        inter_1 = np.matmul(D, JcWcc_inv)
        inter_2 = np.matmul(inter_1, dfree_c + np.matmul(Jc, X_cube) - xc_cube)

        if self.config.nb_robot==3:
            B = inter_2 + np.matmul(np.matmul(D, Jc.T), lambda_c)
        else:
            force = -10
            B = inter_2 + np.matmul(np.matmul(D, Jc.T), [0, force/3, 0, 0, force/3, 0, 0, force/3, 0])

        return B

    def compute_lambda_c_from_dfree_c(self, X_cube, new_X_cube, Wcc, Wca, Jc, new_lambda_a, xc_cube, dfree_c):

        Wcc_inv = np.linalg.inv(Wcc)
        inter_1 = new_X_cube - X_cube
        inter_2 = np.matmul(np.matmul(Wcc_inv, Jc), inter_1)

        inter_3 = np.matmul(np.matmul(Wcc_inv, Wca), new_lambda_a)

        inter_4 = np.matmul(Wcc_inv, dfree_c)

        inter_5 = np.matmul(Wcc_inv, xc_cube)

        new_lambda_c = inter_2 -inter_3 -inter_4 + inter_5

        return new_lambda_c

    def compute_X(self, A, B, new_lambda_a):
        return np.matmul(A, new_lambda_a) + B

    def update_object_position(self, X):
        X_t = X[:3].tolist() + euler_to_quaternion(X[3], X[4], X[5])
        self.root.Cube.MechanicalObject.position.value = [X_t]

    def compute_mask_matrice(self, list_dirs):
        return np.diag(list_dirs)

    def update_goal_position(self):
        goal_pos = self.intermediate_goals[min(self.step // self.waiting_eq_dt - 1, len(self.intermediate_goals) - 1)].tolist()
        self.config.scene_config["goalPos"] = goal_pos
        self.root.Goal.MechanicalObject.position.value = np.array([goal_pos[:3] + euler_to_quaternion(goal_pos[3], goal_pos[4], goal_pos[5])])

    def onAnimateEndEvent(self, event):
        # Manage trajectory
        if self.simu_results:
            eq_counter = self.step % self.waiting_eq_dt
            if eq_counter == 0 and self.step // self.waiting_eq_dt - 1 <= len(self.intermediate_goals) - 1:
                print("[WARNING]  >> Registering trajectory results.")
                self.update_save_data()

                path = "./Results/Trajectories/Data/trajectory_" + self.config.model_name + "_learned.txt" if self.type_use == "learned" else "./Results/Trajectories/Data/trajectory_" + self.config.model_name + "_computed.txt"
                with open(path, 'w') as fp:
                    json.dump(self.save_data, fp)

                path = "./Results/Trajectories/Data/effectors_trajectory_" + self.config.model_name + "_learned.txt" if self.type_use == "learned" else "./Results/Trajectories/Data/effectors_trajectory_" + self.config.model_name + "_computed.txt"
                with open(path, 'w') as fp:
                    json.dump(self.save_data_finger_effectors, fp)



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

    for i in range(n_finger):
        Waa_i = W_t[i * n_constraint: i * n_constraint + n_act_constraint, i * n_constraint: i * n_constraint + n_act_constraint]
        Wca_i = W_t[i * n_constraint + n_act_constraint :(i + 1) * n_constraint,
                i * n_constraint: i * n_constraint + n_act_constraint]
        Wcc_i = W_t[i * n_constraint + n_act_constraint: (i + 1) * n_constraint,
                i * n_constraint + n_act_constraint: (i + 1) * n_constraint]

        list_Wcc.append(Wcc_i)
        list_Wca.append(Wca_i)
        list_Waa.append(Waa_i)

        dfree_a_i = dfree_t[i * n_constraint: i * n_constraint + n_act_constraint]
        list_delta_a_free.append(dfree_a_i)

        dfree_c_i = dfree_t[i * n_constraint + n_act_constraint: (i + 1) * n_constraint]
        list_delta_c_free.append(dfree_c_i)

    return list_Wcc, list_Wca, list_Waa, list_delta_a_free, list_delta_c_free


def compute_mechanical_matrices_from_MLP(MLP_model, n_constraint, scaling, n_act_constraint, s_a_t, W_0_scene, dfree_0_scene, nb_robot, normalization_method):
    """
    Rebuild QP matrices for QP solving using Neural Network

    W = |Waa^1  Wac^1  0      0       0       0    |
        |Wca^1  Wcc^1  0      0       0       0    |
        |0      0      Waa^2  Wac^2   0       0    |
        |0      0      Wca^2  Wcc^2   0       0    |
        |0      0      0      0       Waa^3   Wac^3|
        |0      0      0      0       Wca^3   Wcc^3|
    ----------
    Parameters
    ----------
    MLP_model: neural network
        The neural network we used to predict compliance and dfree value.
    n_constraint: int
        Total number of constraints for one robot
    scaling = list of list of numpy arrays
        Scaling for each component for each matrice
    n_act_constraint: int
        Number of constraints for actuators for one robot
    s_a_t: list of list of float
        Actuation and Contact displacement states for each robot in the scene
    W_0_scene: numpy array
        Compliance matrice projected in constraint space without constraint
    dfree_0_scene: numpy array
        Displacement in free configuration without any actuation
    normalization_method: str
        Method used for normalizing data:
            - None: No normalization method used
            - MinMax: Use minimum and maximum value for normalizing each feature
            - Std: Use mean and standard deviation for normalizing each feature
    ----------
    Outputs
    ----------
    list_Waa: liste of array
        Liste containing W_aa^i for i in [1, 2, 3]
    list_Wca: liste of array
        Liste containing W_ca^i for i in [1, 2, 3]
    list_Wcc: liste of array
        Liste containing W_cc^i for i in [1, 2, 3]
    list_dfree_a: liste of array
        Liste containing delta_a_free^i for i in [1, 2, 3]
    list_dfree_c: list of array
        List containing delta_c_free^i for i in [1, 2, 3]
    """

    def rotate_z_axis(new_Wca, new_Wcc):
        # Rotate Wca
        Rz1 = np.diag([-1,-1])
        for i in range(new_Wca.shape[0]):
            if i % 3 != 2:
                new_Wca[i] = np.matmul(new_Wca[i], Rz1)

        # Rotate Wcc
        Rz1 = create_bloc([np.diag([1,1,-1]) for i in range(3)])
        Rz2 = create_bloc([np.diag([-1,-1,1]) for i in range(3)])
        for i in range(new_Wcc.shape[0]):
            if i % 3 != 2:
                new_Wcc[i] = np.matmul(new_Wcc[i], Rz1)
            else:
                new_Wcc[i] = np.matmul(new_Wcc[i], Rz2)

        return new_Wca, new_Wcc

    def rotate_and_translate_dfree_c(dfree_c, rotate, translate):
        nb_points = int(len(dfree_c)/3)
        points = [dfree_c[3*i:3*(i+1)] for i in range(nb_points)]
        new_points = [np.matmul(rotate, point + translate) for point in points]
        return np.concatenate(new_points, axis = 0)

    def get_dfree(Y):
        dfree = Y[-n_constraint:].detach().numpy()
        if normalization_method == "Std":
            dfree = dfree * (scaling[3][1]) + scaling[2][1] # Rescale dfree
        elif normalization_method == "MinMax":
            dfree = dfree * (scaling[3][1]  - scaling[2][1]) + scaling[2][1] # Rescale dfree
        return dfree

    def rotate_and_translate_dfree_c(dfree_c, rotate, translate):
        nb_points = int(len(dfree_c)/3)
        points = [dfree_c[3*i:3*(i+1)] for i in range(nb_points)]
        new_points = [np.matmul(rotate, point + translate) for point in points]
        return np.concatenate(new_points, axis = 0)

    def get_W_blocks(Y):
        W_pred = Y[:-n_constraint].detach().numpy()
        W = np.zeros((n_constraint, n_constraint))
        W[np.triu_indices(n=n_constraint)] = W_pred
        W[np.tril_indices(n=n_constraint, k=-1)] = W.T[np.tril_indices(n=n_constraint, k=-1)]
        W = W.reshape(-1)
        if normalization_method == "Std":
            W = W * (scaling[3][0].reshape(-1)) + scaling[2][0].reshape(-1) # Rescale W
        elif normalization_method == "MinMax":
            W = W * (scaling[3][0].reshape(-1) - scaling[2][0].reshape(-1)) + scaling[2][0].reshape(-1) # Rescale W
        W = W.reshape(n_constraint,n_constraint)
        Waa = W[0:n_act_constraint, 0:n_act_constraint]
        Wca = W[n_act_constraint:, 0:n_act_constraint]
        Wcc = W[n_act_constraint:, n_act_constraint:]
        return Waa, Wca, Wcc

    def rotate_z_axis(new_Wca, new_Wcc):
        # Rotate Wca
        Rz1 = np.diag([-1, -1])
        for i in range(new_Wca.shape[0]):
            if i % 3 != 2:
                new_Wca[i] = np.matmul(new_Wca[i], Rz1)

        # Rotate Wcc
        Rz1 = create_bloc([np.diag([1, 1, -1]) for i in range(3)])
        Rz2 = create_bloc([np.diag([-1, -1, 1]) for i in range(3)])
        for i in range(new_Wcc.shape[0]):
            if i % 3 != 2:
                new_Wcc[i] = np.matmul(new_Wcc[i], Rz1)
            else:
                new_Wcc[i] = np.matmul(new_Wcc[i], Rz2)
        return new_Wca, new_Wcc


    n_contacts = n_constraint - n_act_constraint
    W_0 = W_0_scene[:n_constraint, :n_constraint]
    dfree_0 = dfree_0_scene[:n_constraint]

    # Prediction
    MLP_lib = importlib.import_module("Libraries.Learning.MLP.learning_tools")

    Y = [torch.tensor(W_0), torch.tensor(dfree_0)]
    X1 = [torch.tensor(W_0), torch.tensor(dfree_0), torch.tensor(s_a_t[0])]

    # Rescale data before providing it to the NN
    if normalization_method == "Std":
        X1, _ = MLP_lib.create_data_std(X1, Y, scaling[0], scaling[1], scaling[2], scaling[3])
    elif normalization_method == "MinMax":
        X1, _ = MLP_lib.create_data_minmax(X1, Y, scaling[0], scaling[1], scaling[2], scaling[3])
    else:
        X1, _ = MLP_lib.create_data(X1, Y)

    Y1 = MLP_model(X1)

    ### Compute d_free_a from prediction ###
    dfree_1 = get_dfree(Y1)
    list_dfree_a= [dfree_1[0:n_act_constraint]]
    list_dfree_c= [dfree_1[n_act_constraint:n_constraint]]

    dist_y, dist_x = -70, -30
    list_dfree_c[0] = rotate_and_translate_dfree_c(list_dfree_c[0], np.diag([1, 1, 1]), np.array([0, 0, 0]))

    ### Compute W from prediction ###
    Waa_1, Wca_1, Wcc_1 = get_W_blocks(Y1)

    list_Waa = [Waa_1]
    list_Wca = [Wca_1]
    list_Wcc = [Wcc_1]

    if nb_robot == 3:
        #Same for the other robots
        X2 = [torch.tensor(W_0), torch.tensor(dfree_0), torch.tensor(s_a_t[1])]
        X3 = [torch.tensor(W_0), torch.tensor(dfree_0), torch.tensor(s_a_t[2])]
        if normalization_method == "Std":
            X2, _ = MLP_lib.create_data_std(X2, Y, scaling[0], scaling[1], scaling[2], scaling[3])
            X3, _ = MLP_lib.create_data_std(X3, Y, scaling[0], scaling[1], scaling[2], scaling[3])
        elif normalization_method == "MinMax":
            X2, _ = MLP_lib.create_data_minmax(X2, Y, scaling[0], scaling[1], scaling[2], scaling[3])
            X3, _ = MLP_lib.create_data_minmax(X3, Y, scaling[0], scaling[1], scaling[2], scaling[3])
        else:
            X2, _ = MLP_lib.create_data(X2, Y)
            X3, _ = MLP_lib.create_data(X3, Y)

        Y2 = MLP_model(X2)
        Y3 = MLP_model(X3)

        dfree_2 = get_dfree(Y2)
        dfree_3 = get_dfree(Y3)

        list_dfree_a+=  [dfree_2[0:n_act_constraint], dfree_3[0:n_act_constraint]]
        list_dfree_c+= [dfree_2[n_act_constraint:n_constraint], dfree_3[n_act_constraint:n_constraint]]

        list_dfree_c[1] = rotate_and_translate_dfree_c(list_dfree_c[1], np.diag([-1, -1, 1]), np.array([dist_x, dist_y, 0]))
        list_dfree_c[2] = rotate_and_translate_dfree_c(list_dfree_c[2], np.diag([-1, -1, 1]), np.array([-dist_x, dist_y, 0]))


        Waa_2, Wca_2, Wcc_2 = get_W_blocks(Y2)
        Waa_3, Wca_3, Wcc_3 = get_W_blocks(Y3)

        # Apply rotation to second and third Finger learned matrices
        # Matrices are rotated aroudn z-axis
        new_Wca_2, new_Wcc_2 = rotate_z_axis(Wca_2, Wcc_2)
        new_Wca_3, new_Wcc_3 = rotate_z_axis(Wca_3, Wcc_3)

        list_Waa += [Waa_2, Waa_3]
        list_Wca += [new_Wca_2, new_Wca_3]
        list_Wcc += [new_Wcc_2, new_Wcc_3]


    return list_Waa, list_Wca, list_Wcc, list_dfree_a, list_dfree_c
