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

import Mesh.Constants as Const
from Libraries.Simulation.BaseController import BaseController
from Sofa import SofaConstraintSolver

class Controller(BaseController):
    """See BaseController for a detailed description.
    """
    def __init__(self, *args, **kwargs):
        super(Controller,self).__init__(*args, **kwargs)
    def get_actuators_state(self):
        return [float(cable.cableInitialLength.value)-float(cable.cableLength.value) for cable in self.list_actuators]
    def get_effectors_state(self):
        effector_state = []
        for effector_MO in self.list_effectors_MO:
            # pos = effector_MO.position.value[0]
            # rest_pos = effector_MO.rest_position.value[0]
            # res = pos - rest_pos
            # effector_state += res.tolist()
            effector_state += [0,0,0]
        return effector_state
    def get_effectors_positions(self):
        return [effector_MO.position.value for effector_MO in self.list_effectors_MO]
    def apply_actions(self, values):
        assert len(self.list_actuators) == len(values)
        for actuator, value in zip(self.list_actuators, values):
            actuator.value.value = [value]

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

def add_goal_node(parent, position = [0, 0, 0], showObject = True):
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
    goal = parent.addChild("Goal")
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

    def __init__(self, parentNode, name = "Finger", youngModulus=3000, poissonRatio=0.3, totalMass=0.1, inverseMode=False, translation = [0, 0, 0], rotation = [0, 0, 0], is_force=False):
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
        self.node.addObject('TetrahedronFEMForceField', template='Vec3d', name='FEM', method='large', poissonRatio=poissonRatio,  youngModulus=youngModulus)

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
        effectors.addObject('ConstraintPoint', template='Vec3', indices = [i for i in range(len(TransformedPosEffectors))], effectorGoal=target.position.getLinkPath(), valueType = "force", imposedValue = [0,0,0])
        effectors.addObject("BarycentricMapping", mapForces=False, mapMasses=False)
        return effectors

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

    rootNode.addObject('BVHNarrowPhase')
    rootNode.addObject('BruteForceBroadPhase')
    rootNode.addObject('RuleBasedContactManager', responseParams="mu="+str(0.8), name='Response',
                           response='FrictionContactConstraint')
    rootNode.addObject('LocalMinDistance', alarmDistance=2, contactDistance=0.1, angleCone=0.01)


    rootNode.gravity.value = [0., 0, -9810.]
    rootNode.dt.value = 0.01

    simulation = rootNode.addChild("Simulation")

    simulation.addObject('EulerImplicitSolver', name='odesolver', firstOrder=0, rayleighMass=0.1,  rayleighStiffness=0.1)
    simulation.addObject('SparseLDLSolver', name='precond', template = "CompressedRowSparseMatrixd")
    simulation.addObject('GenericConstraintCorrection', linearSolver="@./precond")

    finger = Finger(simulation, inverseMode=config["inverseMode"],  is_force=config["is_force"], totalMass = 0.0)

    rootNode.finger = finger
    finger.addVisualModel()

    goal_mo = add_goal_node(rootNode, config["goalPos"])
    if config["inverseMode"]:
        effectors = finger.addEffectors(goal_mo, position = [[0.0, Const.Height/2, -3.0*Const.Length]])
    else:
        effectors = finger.addEffectors(goal_mo, position = [[0.0, Const.Height/2, -3.0*Const.Length]])
        rootNode.addObject(Controller(name="Controller", constraint_solver = constraint_solver, constraint_solver_setter=constraint_solver_setter,list_actuators = finger.cables, list_effectors = [effectors.ConstraintPoint], list_effectors_MO = [effectors.MechanicalObject]))

#    SurfaceMeshPath = MeshPath + 'Floor.stl'
#    floorVisu = rootNode.addChild('floorVisu')
#    floorVisu.addObject('MeshSTLLoader', name='loader', filename = SurfaceMeshPath, translation = [0, 0, 0.5], rotation = [0, 90, 0])
#    floorVisu.addObject('OglModel', src="@loader", scale3d=[1, 1, 1])

    return rootNode
