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

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))
MeshPath = os.path.dirname(os.path.abspath(__file__))+'/Mesh/'

import copy

import Mesh.Constants as Const
from Libraries.Simulation.BaseController import BaseController
from Cube import Cube
from Sofa import SofaConstraintSolver

class Controller(BaseController):
    """See BaseController for a detailed description.
    """
    def __init__(self, *args, **kwargs):
        super(Controller,self).__init__(*args, **kwargs)
        self.list_contacts = kwargs["list_contacts"]
        self.list_cube_contacts = kwargs["list_cube_contacts"]
        self.config = kwargs["config"]

        self.n_eq_dt = self.config.get_n_eq_dt()
        self.n_dt = self.config.get_n_dt()

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
        return [0 for _ in range(n_constraint)]

    def get_effectors_positions(self):
        return [effector_MO.position.value for effector_MO in self.list_effectors_MO]

    def apply_actions(self, value):
        size_value_one_finger = int(len(value)/self.config.nb_robot)
        finger_1_value = value[:size_value_one_finger]
        finger_2_value = value[size_value_one_finger:2*size_value_one_finger]
        finger_3_value = value[2*size_value_one_finger:]

        nb_actuators_one_finger = int(len(self.list_actuators)/self.config.nb_robot)
        cable_value = finger_1_value[:nb_actuators_one_finger] + finger_2_value[:nb_actuators_one_finger] + finger_3_value[:nb_actuators_one_finger]
        contact_values = finger_1_value[nb_actuators_one_finger:] + finger_2_value[nb_actuators_one_finger:] + finger_3_value[nb_actuators_one_finger:]

        for i, actuator in enumerate(self.list_actuators):
            actuator.value.value = [cable_value[i]]
        # for i, contact in enumerate(self.list_contacts):
        #     contact_value = contact_values[3*i:3*(i+1)]
        #     contact.PositionConstraint.imposedValue.value = contact_value

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
                                                    "Sofa.Component.Engine.Generate",
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
        rootNode.addObject(Controller(name="Controller", config = classConfig, constraint_solver = constraint_solver, constraint_solver_setter = constraint_solver_setter, list_actuators = finger1.cables + finger2.cables + finger3.cables, list_contacts = finger1.contacts + finger2.contacts + finger3.contacts, list_effectors = list_effectors, list_effectors_MO = list_effectors_MO, list_cube_contacts = cube.contacts))

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
