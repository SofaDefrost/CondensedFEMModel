# -*- coding: utf-8 -*-
"""Finger.py: create scene of the Finger and the controler to actuate it.
"""

__authors__ = "emenager, tnavez"
__contact__ = "etienne.menager@inria.fr, tanguy.navez@inria.fr"
__version__ = "1.0.0"
__copyright__ = "(c) 2020, Inria"
__date__ = "Jun 29 2022"

import sys
import pathlib
from math import cos, sin
import numpy as np
import os
import Sofa
import copy

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))
MeshPath = os.path.dirname(os.path.abspath(__file__))+'/Mesh/'

import Mesh.Constants as Const
from Libraries.Simulation.BaseController import BaseController
from Cube import Cube
from Sofa import SofaConstraintSolver
from Libraries.Simulation.QPControllers.QPproblem import init_QP, solve_QP
import math

def create_bloc(list_matrices):
    size = list_matrices[0].shape
    block = []
    for i in range(len(list_matrices)):
        line = [np.zeros(size) for _ in range(0, i)] + [list_matrices[i]] + [np.zeros(size) for _ in range(i, len(list_matrices)-1)]
        block.append(line)
    return np.block(block)


class Controller(BaseController):
    """See BaseController for a detailed description.
    """
    def __init__(self, *args, **kwargs):
        super(Controller, self).__init__(*args, **kwargs)
        self.list_contacts = kwargs["list_contacts"]
        self.list_cube_contacts = kwargs["list_cube_contacts"]
        self.config = kwargs["config"]
        self.cube = kwargs["cube"]

        self.list_actuator = kwargs["list_actuators"]

        self.n_eq_dt = self.config.get_n_eq_dt()
        self.n_dt = self.config.get_n_dt()

        self.n_constraint = len(self.config.get_actuators_variables()) + len(self.config.get_contacts_variables())  # Number of constraints for one finger
        self.n_act_constraint = int(len(self.list_actuators) / self.config.nb_robot)  # Number of actuators for one finger
        self.solver = init_QP(self.config.nb_robot * self.n_act_constraint)
        self.step = 0

        self.waiting_time = self.config.get_n_eq_dt()
        self.waiting_eq_dt = self.config.get_n_dt()

    def apply_actions(self, values):
        pass

    def get_actuators_state(self):
        """ Actuator state rotated for sampling purposes """
        nb_act_per_robot = int(len(self.list_actuators)/self.config.nb_robot)
        nb_contact_per_robot = int(3*len(self.list_contacts)/self.config.nb_robot)

        cable = [float(cable.cableLength.value) for cable in self.list_actuators]
        contacts = []

        for i, contact in enumerate(self.list_contacts):
            n_curr_robot = int(i // (nb_contact_per_robot / 3))
            pos = contact.MechanicalObject.position.value[0]
            rest_pos = contact.MechanicalObject.rest_position.value[0]

            if n_curr_robot == 0:
                rot = np.diag([1, 1, 1])
            elif n_curr_robot == 1:
                rot = np.diag([-1, -1, 1])
            elif n_curr_robot == 2:
                rot = np.diag([-1, -1, 1])

            res = np.matmul(rot, pos - rest_pos)

            contacts += res.tolist()

        states = []
        for id_robot in range(self.config.nb_robot):
            state = cable[id_robot*nb_act_per_robot:(id_robot+1)*nb_act_per_robot] + contacts[id_robot*nb_contact_per_robot:(id_robot+1)*nb_contact_per_robot]
            states.append(state)
        return states

    def get_actuators(self):
        """Note: PositionConstraint in list_contact so 3 direction at each PositionConstraint.
        """
        return self.list_actuators + self.list_contacts

    def get_effectors_state(self):
        n_constraint = self.get_n_effectors_constraints()
        return [[0 for _ in range(n_constraint)] for _ in range(self.config.nb_robot)]

    def get_compliance_matrice_in_constraint_space(self):
        W_t = copy.deepcopy(self.constraint_solver.W())
        self.list_Wcc, self.list_Wca, self.list_Waa = self.compute_W_from_simulation(self.config.nb_robot, self.n_constraint, self.n_act_constraint, W_t)

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

        self.list_Wca[1], self.list_Wcc[1] = rotate_z_axis(self.list_Wca[1], self.list_Wcc[1] )
        self.list_Wca[2],  self.list_Wcc[2]  = rotate_z_axis(self.list_Wca[2], self.list_Wcc[2])


        def construct_matrix_W(Waa, Wca, Wcc):
            W = np.zeros(shape = (self.n_constraint, self.n_constraint))
            W[:self.n_act_constraint, :self.n_act_constraint] = Waa
            W[self.n_act_constraint:, :self.n_act_constraint] = Wca
            W[self.n_act_constraint:, self.n_act_constraint:] = Wcc
            W[:self.n_act_constraint, self.n_act_constraint:] = Wca.T
            return W

        W_1 = construct_matrix_W(self.list_Waa[0], self.list_Wca[0], self.list_Wcc[0])
        W_2 = construct_matrix_W(self.list_Waa[1], self.list_Wca[1], self.list_Wcc[1])
        W_3 = construct_matrix_W(self.list_Waa[2], self.list_Wca[2], self.list_Wcc[2])

        return [W_1, W_2, W_3]

    def get_dfree(self):
        dfree_t = copy.deepcopy(self.constraint_solver.dfree())
        self.list_delta_a_free, self.list_delta_c_free = self.compute_dfree_from_simulation(self.config.nb_robot,  self.n_constraint, self.n_act_constraint, dfree_t)
        def rotate_and_translate_dfree_c(dfree_c, rotate, translate):
            nb_points = int(len(dfree_c) / 3)
            points = [dfree_c[3 * i:3 * (i + 1)] for i in range(nb_points)]
            new_points = [np.matmul(rotate, point + translate) for point in points]
            return np.concatenate(new_points, axis=0)

        self.list_delta_c_free[0] = rotate_and_translate_dfree_c(self.list_delta_c_free[0], np.diag([1, 1, 1]), np.array([0, 0, 0]))
        self.list_delta_c_free[1] = rotate_and_translate_dfree_c(self.list_delta_c_free[1], np.diag([-1, -1, 1]), np.array([-30, -70, 0]))
        self.list_delta_c_free[2] = rotate_and_translate_dfree_c(self.list_delta_c_free[2], np.diag([-1, -1, 1]), np.array([30, -70, 0]))

        dfree = []
        for i in range(len(self.list_delta_c_free)):
            dfree.append(np.concatenate([self.list_delta_a_free[i], self.list_delta_c_free[i]], axis = 0))

        return dfree


    def get_effectors_positions(self):
        return [copy.deepcopy(effector_MO.position.value) for effector_MO in self.list_effectors_MO]

    def get_cube_pos(self):
        cube_pos = copy.deepcopy(self.cube.MechanicalObject.position.value)[0]
        cube_trans = cube_pos[:3].tolist()
        [x, y, z, w] = cube_pos[3:].tolist()

        #Convert quaternion to euler
        [roll_x, pitch_y, yaw_z] = self.quaternion_to_euler(x, y, z, w)
        return np.array(cube_trans + [roll_x, pitch_y, yaw_z])

    def get_cube_contact_pos_vector(self):
        list_contact_pos = []
        for contact_point in self.list_cube_contacts:
            list_contact_pos += copy.deepcopy(contact_point.MechanicalObject.position.value)[0, :3].tolist()
        return np.array(list_contact_pos)

    def quaternion_to_euler(self, x, y, z, w):
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)

        return [roll_x, pitch_y, yaw_z]

    def euler_to_quaternion(self, roll, pitch, yaw):

        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

        return [qx, qy, qz, qw]

    def compute_b(self, B, X_goal):
        return B - X_goal

    def compute_D(self, Jc, Wcc):
        Wcc_inv = np.linalg.inv(Wcc)
        inter_1 = np.matmul(Wcc_inv, Jc)
        res = np.matmul(Jc.T, inter_1)
        return np.linalg.inv(res)

    def compute_Cik(self):
        pos = []
        for contact in self.list_cube_contacts:
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

        return inter_2

    def compute_QP_matrices(self, A, b, delta_free_a, Waa, lambda_a):

        H = 2 * np.matmul(A.T, A)
        g = 2 * np.matmul(A.T, b)
        C = Waa

        ### Actuation constraints
        l = np.zeros(
            lambda_a.shape[0]) - np.infty  # + 0.0 # Actuation should be more than 0 - We can't push with a cable
        u = np.zeros(lambda_a.shape[0]) + np.infty

        lb = (np.array([-np.inf for i in range(len(delta_free_a))])).astype('double')
        ub = (np.array([np.inf for i in range(len(delta_free_a))])).astype('double')

        return H, g, C, l, u, lb, ub

    def compute_X(self, A, B, new_lambda_a):
        return np.matmul(A, new_lambda_a) + B
    def compute_mask_matrice(self, list_dirs):
        return np.diag(list_dirs)

    def compute_W_from_simulation(self, n_finger, n_constraint, n_act_constraint, W_t):
        list_Wcc, list_Wca, list_Waa = [], [], []

        n_contact = n_constraint - n_act_constraint
        slide = 3 * n_contact
        for i in range(n_finger):
            Waa_i = W_t[slide + i * n_constraint: slide + i * n_constraint + n_act_constraint,
                    slide + i * n_constraint:slide + i * n_constraint + n_act_constraint]
            Wca_i = W_t[slide + i * n_constraint + n_act_constraint:slide + (i + 1) * n_constraint,
                    slide + i * n_constraint: slide + i * n_constraint + n_act_constraint]
            Wcc_i = W_t[slide + i * n_constraint + n_act_constraint: slide + (i + 1) * n_constraint,
                    slide + i * n_constraint + n_act_constraint: slide + (i + 1) * n_constraint]

            list_Wcc.append(Wcc_i)
            list_Wca.append(Wca_i)
            list_Waa.append(Waa_i)

        return list_Wcc, list_Wca, list_Waa

    def compute_dfree_from_simulation(self, n_finger, n_constraint, n_act_constraint, dfree_t):
        list_delta_a_free, list_delta_c_free = [], []

        n_contact = n_constraint - n_act_constraint
        slide = 3 * n_contact
        for i in range(n_finger):
            dfree_a_i = dfree_t[slide + i * n_constraint: slide + i * n_constraint + n_act_constraint]
            list_delta_a_free.append(dfree_a_i)

            dfree_c_i = dfree_t[slide + i * n_constraint + n_act_constraint: slide + (i + 1) * n_constraint]
            list_delta_c_free.append(dfree_c_i)

        return  list_delta_a_free, list_delta_c_free

    def compute_lambda_c_from_dfree_c(self, X_cube, new_X_cube, Wcc, Wca, Jc, new_lambda_a, xc_cube, dfree_c):
        Wcc_inv = np.linalg.inv(Wcc)
        inter_1 = new_X_cube - X_cube
        inter_2 = np.matmul(np.matmul(Wcc_inv, Jc), inter_1)
        inter_3 = np.matmul(np.matmul(Wcc_inv, Wca), new_lambda_a)
        inter_4 = np.matmul(Wcc_inv, dfree_c)
        inter_5 = np.matmul(Wcc_inv, xc_cube)
        new_lambda_c = inter_2 - inter_3 - inter_4 + inter_5

        return new_lambda_c

    def onAnimateBeginEvent(self, event):
        if self.step == self.waiting_time:
            self.lambda_a = np.zeros(self.n_act_constraint * self.config.nb_robot)
            self.lambda_c = np.zeros((self.n_constraint - self.n_act_constraint) * self.config.nb_robot)

        if self.step < self.waiting_time:
            print("[INFO]  >>  Waiting time {}/{}".format(self.step, self.waiting_time))
        else:
            W_t = copy.deepcopy(self.constraint_solver.W())
            dfree_t = copy.deepcopy(self.constraint_solver.dfree())

            self.list_Wcc, self.list_Wca, self.list_Waa =  self.compute_W_from_simulation(self.config.nb_robot, self.n_constraint, self.n_act_constraint, W_t)
            self.list_delta_a_free, self.list_delta_c_free = self.compute_dfree_from_simulation(self.config.nb_robot, self.n_constraint, self.n_act_constraint, dfree_t)

            X_cube = self.get_cube_pos()
            self.X_goal = np.array(self.config.scene_config["goalPos"])
            xc_cube = self.get_cube_contact_pos_vector()

            Cik = self.compute_Cik().tolist()
            Jc = self.compute_Jc(Cik)
            Wcc, Wca, Waa, delta_free_a = self.compute_matrices()


            D = self.compute_D(Jc, Wcc)
            A = self.compute_A(Jc, Wcc, Wca, D)

            dfree_c = np.concatenate(self.list_delta_c_free, axis=0)
            dfree_a = np.concatenate(self.list_delta_a_free, axis=0)

            B = self.compute_B_from_dfree_c(X_cube, self.lambda_c, D, Jc, Wcc, dfree_c, xc_cube)
            b = self.compute_b(B, self.X_goal)

            MASK_ROTATION = True
            if MASK_ROTATION:
                list_dirs = [1, 1, 1, 0, 0, 0]
                mask_matrix = self.compute_mask_matrice(list_dirs)
                masked_A = np.matmul(mask_matrix, A)
                masked_b = np.matmul(mask_matrix, b)
                H, g, C, l, u, lb, ub = self.compute_QP_matrices(masked_A, masked_b, delta_free_a, Waa, self.lambda_a)
            else:
                H, g, C, l, u, lb, ub = self.compute_QP_matrices(A,  b, delta_free_a, Waa, self.lambda_a)

            # Solve QP
            new_lambda_a = solve_QP(self.solver, H, g, C, l, u, lb, ub, is_init=(self.step != self.waiting_time))

            factor = 1
            self.lambda_a = self.lambda_a + factor * (new_lambda_a - self.lambda_a)

            new_X_cube = self.compute_X(A, B, self.lambda_a)
            self.lambda_c = self.compute_lambda_c_from_dfree_c(X_cube, new_X_cube, Wcc, Wca, Jc, self.lambda_a, xc_cube, dfree_c)

            delta_a = np.matmul(Waa, self.lambda_a) + np.matmul(Wca.T, self.lambda_c) + dfree_a

            # print("lambda_a: ", self.lambda_a)
            # print("lambda_c: ", self.lambda_c)
            # print("delta_a:  ", delta_a)

            for i, actuator in enumerate(self.list_actuator):
                actuator.value.value = [delta_a[i]]

        self.step += 1


def transform(points, translation, rotation):
    """ Transform a list of points by applying translation and rotation around axes by given angles.

    Parameters
    ----------
        points: list of numpy arrays
            List of 3D points to transform
        translation: array
            Translation to apply to points
        rotation: list
            Rotation angle (Euler angles) for each axis in degrees
    """

    # Compute rotation matrice
    def Rx(theta):
        return np.matrix([[ 1, 0           , 0           ],
                          [ 0, np.cos(theta),-np.sin(theta)],
                          [ 0, np.sin(theta), np.cos(theta)]])

    def Ry(theta):
        return np.matrix([[ np.cos(theta), 0, np.sin(theta)],
                         [ 0           , 1, 0           ],
                         [-np.sin(theta), 0, np.cos(theta)]])

    def Rz(theta):
        return np.matrix([[ np.cos(theta), -np.sin(theta), 0 ],
                           [ np.sin(theta), np.cos(theta) , 0 ],
                           [ 0           , 0            , 1 ]])


    Rx = Rx(np.deg2rad(rotation[0]))
    Ry = Ry(np.deg2rad(rotation[1]))
    Rz = Rz(np.deg2rad(rotation[2]))
    R = Rz * Ry * Rx

    # Apply transformation
    rt_points = []
    for point in points:
        rt_point = np.dot(R, point) + translation #Rotation and translation
        rt_points.append(rt_point)

    return rt_points

def add_goal_node(parent, name = "Goal", position = [0, 0, 0], showObject = True):
    """Add goal node and MO in the scene.
    Parameters:
    ----------
        parent: Sofa.Node
            The parent of the goal node in scene tree.
        position: list of loat or list of list of float
            The position(s) of the goal.
        showObject: bool
            Flag to indicate if we want to display the goals.
    Outputs:
    -------
        goal_mo: Sofa.MechanicalObject
            The MechanicalObject of the goal.
    """
    goal = parent.addChild(name)
    goal_mo = goal.addObject('MechanicalObject', name='GoalMO', showObject=showObject, drawMode=1, showObjectScale=2,
                             showColor=[0, 1, 0, 1], position=position)
    return goal_mo

class Finger():
    """ This class is implementing a soft robot inspired by a finger.
        The robot is entirely soft and actuated with 2 cables.
        Parameters:
        -----------
            youngModulus: float
                The Young Modulus of the Trunk.
            poissonRatio: float
                The Poisson ratio of the Trunk.
            totalMass: float
                The total mass of the Trunk.
            inverseMode: bool
                If we use the Trunk in inverse mode.
    """

    def __init__(self, parentNode, name = "Finger", youngModulus=100, poissonRatio=0.3, totalMass=0.1, inverseMode=False, translation = [0, 0, 0], rotation = [0, 0, 0], is_force=False):
        """Classical initialization of the class.

        Parameters:
        -----------
            parentNode: Sofa.Node
                The parent of the Trunk node in scene tree.
            name: string
                The name of the object.
            youngModulus: float
                The Young Modulus of the Trunk.
            poissonRatio: float
                The Poisson ratio of the Trunk.
            totalMass: float
                The total mass of the Trunk.
            inverseMode: bool
                If we use the Trunk in inverse mode.
            is_force: bool
                Wether actuators are controlled in force or displacement.

        """

        self.translation = np.array(translation)
        self.rotation = np.array(rotation)

        self.inverseMode = inverseMode
        self.is_force=is_force
        self.node = parentNode.addChild(name)

        VolumetricMeshPath = MeshPath + 'Finger_Volumetric.vtk'
        self.node.addObject('MeshVTKLoader', name='loader', filename=VolumetricMeshPath, scale3d=[1, 1, 1],  translation = self.translation, rotation = self.rotation)
        self.node.addObject('TetrahedronSetTopologyContainer', src='@loader', name='container')
        self.node.addObject('TetrahedronSetGeometryAlgorithms')

        self.node.addObject('MechanicalObject', name='dofs', template='Vec3d', showIndices='false', showIndicesScale='4e-5')
        self.node.addObject('UniformMass', totalMass=totalMass)
        self.node.addObject('TetrahedronFEMForceField', template='Vec3d', name='FEM', method='small', poissonRatio=poissonRatio,  youngModulus=youngModulus)

        BoxMargin = 3
        BoxCoordP1 = np.array([-(Const.Thickness/2+BoxMargin), -BoxMargin, 0])
        BoxCoordP2 = np.array([-(Const.Thickness/2+BoxMargin), Const.Height+2*BoxMargin, 0])
        BoxCoordP3 = np.array([Const.Thickness/2+BoxMargin,Const.Height+2*BoxMargin, 0])
        TransformedBoxCoords = transform([BoxCoordP1, BoxCoordP2, BoxCoordP3], self.translation, self.rotation)
        BoxCoords = np.concatenate((TransformedBoxCoords[0].flatten(), TransformedBoxCoords[1].flatten(), TransformedBoxCoords[2].flatten(), np.array([[2*BoxMargin]])), axis=1)
        self.node.addObject('BoxROI', name='boxROI', orientedBox=BoxCoords, drawBoxes=True)
        self.node.addObject('RestShapeSpringsForceField', points='@boxROI.indices', stiffness=1e10)

        self.cables = self._addCables()

    def _addCables(self, BellowGap = 0):
        """Private method to add cables in the Finger.
        """
         # Cable Actuation
        cables_node = self.node.addChild('cables')
        cables = []

        NSegments = 3
        CableHeight = 2*(Const.Height-Const.JointHeight)/3
        LengthDiagonal = CableHeight/np.cos(Const.JointSlopeAngle)
        JointStandoff = LengthDiagonal*np.sin(Const.JointSlopeAngle)
        CableDistance = Const.CableDistance
        BellowGap = BellowGap

        valueType = "force" if self.is_force else "displacement"

        # Cable 1
        CablePoints = []
        pullPoint = transform([np.array([-CableDistance/2, CableHeight+Const.JointHeight, 0])], self.translation, self.rotation)
        for i in range(NSegments):
            SegmentOffsetBase = (Const.Length+BellowGap)*i
            SegmentOffsetTip  = Const.Length*(i+1)+BellowGap*i
            CablePoints.append([-CableDistance/2,CableHeight+Const.JointHeight,-JointStandoff - SegmentOffsetBase])
            CablePoints.append([-CableDistance/2,CableHeight+Const.JointHeight, JointStandoff - SegmentOffsetTip])
        TransformedCablePoints = transform(np.array(CablePoints), self.translation, self.rotation)

        cable1 = cables_node.addChild('cable1')
        cable1.addObject('MechanicalObject', position=TransformedCablePoints)
        if self.inverseMode:
            cable = cable1.addObject('CableActuator', template='Vec3d', name='CableConstraint', indices=list(range(2*NSegments)), pullPoint= pullPoint, printLog=True)
        else:
            cable = cable1.addObject('CableConstraint', template='Vec3d', name='CableConstraint', indices=list(range(2*NSegments)), pullPoint= pullPoint, printLog=True, valueType = valueType)
        cable1.addObject('BarycentricMapping')
        cables.append(cable)

        # Cable 2
        CablePoints = []
        pullPoint = transform([np.array([CableDistance/2, CableHeight+Const.JointHeight, 0])], self.translation, self.rotation)
        for i in range(NSegments):
            SegmentOffsetBase = (Const.Length+BellowGap)*i
            SegmentOffsetTip  = Const.Length*(i+1)+BellowGap*i
            CablePoints.append([CableDistance/2,CableHeight+Const.JointHeight,-JointStandoff - SegmentOffsetBase])
            CablePoints.append([CableDistance/2,CableHeight+Const.JointHeight, JointStandoff - SegmentOffsetTip])
        TransformedCablePoints = transform(np.array(CablePoints), self.translation, self.rotation)

        cable2 = cables_node.addChild('cable2')
        cable2.addObject('MechanicalObject', position=TransformedCablePoints)
        if self.inverseMode:
            cable = cable2.addObject('CableActuator', template='Vec3d', name='CableConstraint', indices=list(range(2*NSegments)), pullPoint= pullPoint, printLog=True)
        else:
            cable = cable2.addObject('CableConstraint', template='Vec3d', name='CableConstraint', indices=list(range(2*NSegments)), pullPoint= pullPoint, printLog=True, valueType = valueType)
        cable2.addObject('BarycentricMapping')
        cables.append(cable)

        return cables

    def addVisualModel(self):
        """Add a visual model of the Finger.
        """
        SurfaceMeshPath = MeshPath + 'Finger_Surface.stl'
        modelVisu = self.node.addChild('visu')
        modelVisu.addObject('MeshSTLLoader', filename=SurfaceMeshPath, name="loader", translation = self.translation, rotation = self.rotation)
        modelVisu.addObject('OglModel', src="@loader", scale3d=[1, 1, 1])
        modelVisu.addObject('BarycentricMapping')

    def addEffectors(self, target, position=[[0.0, Const.Height/2, -3.0*Const.Length]]):
        """Add a position effector in the Finger.
        Parameters:
        ----------
            target:
            position: list of float or list of list of float
                The position of the effector(s) in the Finger.
        """
        effectors = self.node.addChild("Effectors")
        TransformedPosEffectors = transform(position, self.translation, self.rotation)
        effectors.addObject("MechanicalObject", position=TransformedPosEffectors)
        effectors.addObject('ConstraintPoint', template='Vec3', indices = [i for i in range(len(TransformedPosEffectors))])
        effectors.addObject("BarycentricMapping", mapForces=False, mapMasses=False)
        return effectors

    def addContact(self, nb_points = 3, position_center= [0.0, Const.Height/2, -3.0*Const.Length], r = 1, use_collision = False, group = 0):
        """Add contact point on the Finger.
        The contact point are disposed on a circle of radius r and center position_center,
        -----------
        Inputs:
        -----------
            nb_points: int
                Number of contact points
            position_center: list
                The center of the circle.
            r: float
                the radius of the circle.
            use_collision: bool
                If we use collision model for the mechanical object
        -----------
        Outputs:
        -----------
            pos_contacts: list of numpy array
                List of positions for contact points
        """
        self.contacts = []
        contacts = self.node.addChild("Contacts")
        ang = np.pi/nb_points
        pos_contacts = []
        for i in range(nb_points):
            position = [[position_center[0]+r*np.sin(2*i*ang), position_center[1], position_center[2]+r*np.cos(2*i*ang)]]
            TransformedPosContacts = transform(position, self.translation, self.rotation)
            pos_contacts.append(TransformedPosContacts)
            _contact = contacts.addChild("Contact_"+str(i))
            _contact.addObject("MechanicalObject", position=TransformedPosContacts, showObject = True, showObjectScale = 10, showColor = "green")
            if use_collision:
                _contact.addObject('PointCollisionModel', group=group)
            _contact.addObject('ConstraintPoint', template='Vec3', indices = [0], valueType = "force", imposedValue = [0,0,0])
            _contact.addObject("BarycentricMapping", mapForces=False, mapMasses=False, input = "@../../")
            self.contacts.append(_contact)
        return pos_contacts


def createScene(rootNode, classConfig):
    config = classConfig.get_scene_config()
    rootNode.addObject("RequiredPlugin", name="SoftRobots")
    rootNode.addObject("RequiredPlugin", name="SofaPython3")
    rootNode.addObject("RequiredPlugin", name="CondensedFEMModel")

    rootNode.addObject("RequiredPlugin", pluginName=["Sofa.Component.AnimationLoop",
                                                    "Sofa.GL.Component.Rendering2D",
                                                    "Sofa.GL.Component.Rendering3D",
                                                    "Sofa.GL.Component.Shader",
                                                    "Sofa.Component.Collision.Geometry",
                                                    "Sofa.Component.Collision.Detection.Intersection",
                                                    "Sofa.Component.Collision.Response.Contact",
                                                    "Sofa.Component.MechanicalLoad",
                                                    "Sofa.Component.Engine.Select",
                                                    "Sofa.Component.Diffusion",
                                                    "Sofa.Component.SolidMechanics.FEM.Elastic",
                                                    "Sofa.Component.SolidMechanics.Spring",
                                                    "Sofa.Component.Playback",
                                                    'Sofa.Component.Collision.Detection.Algorithm',
                                                    "Sofa.Component.LinearSolver.Preconditioner",
                                                     "Sofa.Component.LinearSolver.Iterative",
                                                     "Sofa.Component.Constraint.Lagrangian.Correction",
                                                     "Sofa.Component.Constraint.Lagrangian.Model",
                                                     "Sofa.Component.Constraint.Lagrangian.Solver",
                                                     "Sofa.Component.Constraint.Projective",
                                                     "Sofa.Component.IO.Mesh",
                                                     "Sofa.Component.LinearSolver.Direct",
                                                     "Sofa.Component.Mapping.MappedMatrix",
                                                     "Sofa.Component.Mass",
                                                     "Sofa.Component.ODESolver.Backward",
                                                     "Sofa.Component.Topology.Container.Dynamic",
                                                 ])
    source = config["source"]
    target = config["target"]

    rootNode.addObject("LightManager")
    spotLoc = [2*source[0], 0, 0]
    rootNode.addObject("SpotLight", position=spotLoc, direction=[-np.sign(source[0]), 0.0, 0.0])
    rootNode.addObject("InteractiveCamera", name='camera', position=source, lookAt=target, zFar=500)
    rootNode.addObject('VisualStyle', displayFlags='showVisualModels hideCollisionModels hideForceFields')

    rootNode.addObject("DefaultVisualManagerLoop")

    rootNode.addObject('DefaultPipeline')
    rootNode.addObject('FreeMotionAnimationLoop')

    if config["inverseMode"]:
        rootNode.addObject('RequiredPlugin', name='SoftRobots.Inverse')
        rootNode.addObject("QPInverseProblemSolver")
    else:
        constraint_solver = rootNode.addObject('GenericConstraintSolver', tolerance=1e-6, maxIterations=1000)
        constraint_solver_setter = rootNode.addObject("ConstraintSolverSetter")

    rootNode.gravity.value = [0., 0, -9810.]
    rootNode.dt.value = 0.01

    simulation = rootNode.addChild("Simulation")

    simulation.addObject('EulerImplicitSolver', firstOrder=1, rayleighMass=0.1,  rayleighStiffness=0.1)
    simulation.addObject('SparseLDLSolver', template = "CompressedRowSparseMatrixd")
    simulation.addObject('GenericConstraintCorrection', linearSolver="@./SparseLDLSolver")


    #Fingers
    dist_y, dist_x = 70, 30
    config["is_force"] = False
    finger1 = Finger(simulation, name = "Finger1", inverseMode=config["inverseMode"],  is_force=config["is_force"], totalMass = 0.0, translation = [0, 0, 0], rotation = [0, 0, 0])
    finger2 = Finger(simulation, name = "Finger2", inverseMode=config["inverseMode"],  is_force=config["is_force"], totalMass = 0.0, translation = [dist_x, dist_y, 0], rotation = [0, 0, 180])
    finger3 = Finger(simulation, name = "Finger3", inverseMode=config["inverseMode"],  is_force=config["is_force"], totalMass = 0.0, translation = [-dist_x, dist_y, 0], rotation = [0, 0, 180])

    finger1.addVisualModel()
    finger2.addVisualModel()
    finger3.addVisualModel()

    rootNode.finger1 = finger1
    rootNode.finger2 = finger2
    rootNode.finger3 = finger3

    list_effectors = []
    list_effectors_MO = []

    pos_contacts_1 = finger1.addContact(nb_points = config["nb_contact_points"], position_center=[0.0, Const.Height, -2.5*Const.Length], r = config["contact_radius"], use_collision = False, group=0)
    pos_contacts_2 = finger2.addContact(nb_points = config["nb_contact_points"], position_center=[0.0, Const.Height, -2.5*Const.Length], r = config["contact_radius"], use_collision = False, group=0)
    pos_contacts_3 = finger3.addContact(nb_points = config["nb_contact_points"], position_center=[0.0, Const.Height, -2.5*Const.Length], r = config["contact_radius"], use_collision = False, group=0)

    #Cube
    cube_config = {"init_pos": [0, dist_y/2, -3.0*Const.Length+25], "scale": [dist_y/2, dist_y/8+5, dist_y/8+5], "density": 0.0000001}
    cube = Cube(name = "Cube", cube_config=cube_config)
    cube.onEnd(simulation, collisionGroup=1, withSolver = False)
    cube.addContact(pos_contacts_1 + pos_contacts_2 + pos_contacts_3)


    if not config["inverseMode"]:
        rootNode.addObject(
            Controller(name="Controller", cube=cube.cube, config=classConfig, constraint_solver=constraint_solver,
                        constraint_solver_setter = constraint_solver_setter,
                       list_actuators=finger1.cables + finger2.cables + finger3.cables,
                       list_contacts=finger1.contacts + finger2.contacts + finger3.contacts,
                       list_effectors=list_effectors, list_effectors_MO=list_effectors_MO,
                       list_cube_contacts=cube.contacts))

    ### Goal
    Goal = rootNode.addChild("Goal")
    goal_pos = config["goalPos"]
    goal_quat = euler_to_quaternion(goal_pos[3], goal_pos[4], goal_pos[5])
    Goal.addObject("MechanicalObject", template='Rigid3', position=goal_pos[:3] + goal_quat, showObject=True, showObjectScale=5, drawMode=0)

    for i, contact in enumerate(finger1.contacts + finger2.contacts + finger3.contacts):
        mo_cube = cube.contacts[i].MechanicalObject
        simulation.addObject('BilateralInteractionConstraint', name='ContactConstraint_' + str(i),
                       template='Vec3d', object1=contact.MechanicalObject.getLinkPath(),
                       object2=mo_cube.getLinkPath(), first_point = 0, second_point = 0)

    return rootNode


def euler_to_quaternion(roll, pitch, yaw):

    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

    return [qx, qy, qz, qw]
