# -*- coding: utf-8 -*-
"""Trunk.py: create scene of the Trunk and the controler to actuate it.
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


sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))
MeshPath = os.path.dirname(os.path.abspath(__file__))+'/Mesh/'

from Libraries.Simulation.BaseController import BaseController
from Sofa import SofaConstraintSolver

class Controller(BaseController):
    """See BaseController for a detailed description.
    """
    def __init__(self, *args, **kwargs):
        super(Controller,self).__init__(*args, **kwargs)
    def get_actuators_state(self):
        return [float(cable.cableLength.value) for cable in self.list_actuators]
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
                             showColor=[1, 1, 1, 1], position=position)
    return goal_mo

class Trunk():
    """ This class is implementing a soft robot inspired by the elephant's trunk.
        The robot is entirely soft and actuated with 8 cables. This robot is composed
        of three elements:
            - a visual model
            - a collision model
            - a mechanical model for the deformable structure
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

    def __init__(self, parentNode, youngModulus=450, poissonRatio=0.45, totalMass=0.042, inverseMode=False, is_force=False):
        """Classical initialization of the class.

        Parameters:
        -----------
            parentNode: Sofa.Node
                The parent of the Trunk node in scene tree.
            youngModulus: float
                The Young Modulus of the Trunk.
            poissonRatio: float
                The Poisson ratio of the Trunk.
            totalMass: float
                The total mass of the Trunk.
            inverseMode: bool
                If we use the Trunk in inverse mode.

        """

        self.inverseMode = inverseMode
        self.is_force=is_force
        self.node = parentNode.addChild('Trunk')

        self.node.addObject('MeshVTKLoader', name='loader', filename=MeshPath+'trunk.vtk')
        self.node.addObject('TetrahedronSetTopologyContainer', src='@loader', name='container')
        self.node.addObject('TetrahedronSetTopologyModifier')
        self.node.addObject('TetrahedronSetGeometryAlgorithms')

        self.node.addObject('MechanicalObject', name='dofs', template='Vec3d', showIndices='false',
                            showIndicesScale='4e-5')
        self.node.addObject('UniformMass', totalMass=totalMass)
        self.node.addObject('TetrahedronFEMForceField', template='Vec3d', name='FEM', method='large',
                            poissonRatio=poissonRatio,  youngModulus=youngModulus)

        self.cables = self._addCables()

    def _rotate(self, v,q):
        """Private methode to rotate a 3D vector with a quaternion.
        Parameters:
        ----------
            v: list
                The vector we want to rotate.
            q: list
                The quaternion that represents the rotation.
        """
        c0 = ((1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]))*v[0] + (2.0 * (q[0] * q[1] - q[2] * q[3])) * v[1] + (2.0 * (q[2] * q[0] + q[1] * q[3])) * v[2])
        c1 = ((2.0 * (q[0] * q[1] + q[2] * q[3]))*v[0] + (1.0 - 2.0 * (q[2] * q[2] + q[0] * q[0]))*v[1] + (2.0 * (q[1] * q[2] - q[0] * q[3]))*v[2])
        c2 = ((2.0 * (q[2] * q[0] - q[1] * q[3]))*v[0] + (2.0 * (q[1] * q[2] + q[0] * q[3]))*v[1] + (1.0 - 2.0 * (q[1] * q[1] + q[0] * q[0]))*v[2])

        return [c0, c1, c2]

    def _normalize(self, x):
        """Private methode to normalize a 3D vector.
        Parameters:
        ----------
            x: list
                The vector we want to normalize.
        """
        norm = np.sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2])
        for i in range(0,3):
            x[i] = x[i]/norm

    def _addCables(self):
        """Private method to add cables in the Trunk.
        """
        length1 = 10
        length2 = 2
        lengthTrunk = 195
        cables = []
        pullPoint = [[0., length1, 0.], [-length1, 0., 0.], [0., -length1, 0.], [length1, 0., 0.]]
        direction = [0, length2-length1, lengthTrunk]
        self._normalize(direction)

        displacementL = [7.62, -18.1, 3.76, 30.29]
        displacementS = [-0.22, -7.97, 3.89, 12.03]

        nbCables = 4
        valueType = "force" if self.is_force else "displacement"

        for i in range(0,nbCables):
            theta = 1.57*i
            q = [0.,0.,sin(theta/2.), cos(theta/2.)]

            position = [[0, 0, 0]]*20
            for k in range(0,20,2):
                v = [direction[0], direction[1]*17.5*(k/2)+length1, direction[2]*17.5*(k/2)+21]
                position[k] = self._rotate(v,q)
                v = [direction[0], direction[1]*17.5*(k/2)+length1, direction[2]*17.5*(k/2)+27]
                position[k+1] = self._rotate(v,q)

            pullPointList = [[pullPoint[i][0], pullPoint[i][1], pullPoint[i][2]]]

            cableL = self.node.addChild('cableL'+str(i))
            cableL.addObject('MechanicalObject', name='meca',position= pullPointList+ position)

            idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

            if self.inverseMode:
                cable = cableL.addObject('CableActuator', template='Vec3d', name="cable", hasPullPoint=0, indices=idx, maxPositiveDisp=70, maxDispVariation=1, minForce=0)
            else:
                cable = cableL.addObject('CableConstraint', template='Vec3d', name="cable", hasPullPoint=0, indices=idx, maxPositiveDisp=70, maxDispVariation=1, minForce=0, valueType = valueType)
            cableL.addObject('BarycentricMapping', name='mapping',  mapForces=False, mapMasses=False)
            cables.append(cable)
            # pipes
            pipes = self.node.addChild('pipes'+str(i))
            pipes.addObject('EdgeSetTopologyContainer', position= pullPointList + position, edges= [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
            pipes.addObject('MechanicalObject', name="pipesMO")
            pipes.addObject('UniformMass', totalMass=0.003)
            pipes.addObject('MeshSpringForceField', stiffness=1.5e2, damping=0, name="FF")
            pipes.addObject('BarycentricMapping', name="BM")


        for i in range(0,nbCables):
            theta = 1.57*i
            q = [0.,0.,sin(theta/2.), cos(theta/2.)]

            position = [[0, 0, 0]]*10
            for k in range(0,9,2):
                v = [direction[0], direction[1]*17.5*(k/2)+length1, direction[2]*17.5*(k/2)+21]
                position[k] = self._rotate(v,q)
                v = [direction[0], direction[1]*17.5*(k/2)+length1, direction[2]*17.5*(k/2)+27]
                position[k+1] = self._rotate(v,q)

            pullPointList = [[pullPoint[i][0], pullPoint[i][1], pullPoint[i][2]]]

            cableS = self.node.addChild('cableS'+str(i))
            cableS.addObject('MechanicalObject', name='meca', position=pullPointList+ position)

            idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            if self.inverseMode:
                cable = cableS.addObject('CableActuator', template='Vec3d', name="cable", hasPullPoint=0, indices=idx, maxPositiveDisp=40, maxDispVariation=1, minForce=0)
            else:
                cable = cableS.addObject('CableConstraint', template='Vec3d', name="cable", hasPullPoint=0, indices=idx, maxPositiveDisp=40, maxDispVariation=1, minForce=0, valueType = valueType)
            cables.append(cable)
            cableS.addObject('BarycentricMapping', name='mapping',  mapForces=False, mapMasses=False)

        return cables

    def addVisualModel(self, color=[1., 1., 1., 1.]):
        """Add a visual model of the Trunk.
        Parameters:
        ----------
            color: list of 4 floats in [0, 1]
                The color of the visual model (description: RGB + transparency).
        """
        trunkVisu = self.node.addChild('VisualModel')
        trunkVisu.addObject('MeshSTLLoader', name="loader", filename=MeshPath+"trunk.stl")
        trunkVisu.addObject('OglModel', template='Vec3d', src="@loader", color=color)
        trunkVisu.addObject('BarycentricMapping')


    def addCollisionModel(self, selfCollision=False):
        """Add a collision model of the Trunk.
        Parameters:
        ----------
            selfCollision: bool
                Allow the self collision.
        """
        trunkColli = self.node.addChild('CollisionModel')
        for i in range(2):
            part = trunkColli.addChild("Part"+str(i+1))
            part.addObject('MeshSTLLoader', name="loader", filename=MeshPath+"trunk_colli"+str(i+1)+".stl")
            part.addObject('MeshTopology', src="@loader")
            part.addObject('MechanicalObject')
            part.addObject('TTriangleModel', group=1 if not selfCollision else i)
            part.addObject('TLineModel', group=1 if not selfCollision else i)
            part.addObject('TPointModel', group=1 if not selfCollision else i)
            part.addObject('BarycentricMapping')

    def fixExtremity(self):
        """Fix the base of the Trunk.
        """
        self.node.addObject('BoxROI', name='boxROI', box=[[-20, -20, 0], [20, 20, 20]], drawBoxes=False)
        self.node.addObject('PartialFixedConstraint', fixedDirections=[1, 1, 1], indices="@boxROI.indices")

    def addEffectors(self, target, position=[[0., 0., 195.]]):
        """Add a position effector in the Trunk.
        Parameters:
        ----------
            target:
            position: list of float or list of list of float
                The position of the effector(s) in the Trunk.
        """
        effectors = self.node.addChild("Effectors")
        effectors.addObject("MechanicalObject", position=position)
        effectors.addObject('PositionEffector', template='Vec3', indices = [i for i in range(len(position))], effectorGoal=target.position.getLinkPath())
        effectors.addObject("BarycentricMapping", mapForces=False, mapMasses=False)

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
    rootNode.addObject('VisualStyle', displayFlags='showCollision showVisualModels showForceFields '
                                                   'showInteractionForceFields hideCollisionModels '
                                                   'hideBoundingCollisionModels hideWireframe')

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
    rootNode.addObject('RuleBasedContactManager', responseParams="mu="+str(0.3), name='Response',
                           response='FrictionContactConstraint')
    rootNode.addObject('LocalMinDistance', alarmDistance=10, contactDistance=5, angleCone=0.01)


    rootNode.gravity.value = [0., -9810., 0.]
    rootNode.dt.value = 0.01

    simulation = rootNode.addChild("Simulation")
    simulation.addObject('EulerImplicitSolver', name='odesolver', firstOrder=0, rayleighMass=0.1,  rayleighStiffness=0.1)
    simulation.addObject('SparseLDLSolver', name='precond', template = "CompressedRowSparseMatrixd")
    simulation.addObject('GenericConstraintCorrection', linearSolver="@./precond")

    trunk = Trunk(simulation, inverseMode=config["inverseMode"],  is_force=config["is_force"], totalMass = 0.0)

    rootNode.trunk = trunk

    #trunk.addVisualModel(color=[1., 1., 1., 1])
    trunk.fixExtremity()

    goal_mo = add_goal_node(rootNode, position = config["goalPos"], showObject = True)
    if config["inverseMode"]:
        trunk.addEffectors(goal_mo, position = [[0., 0., 195.]])
    else:
        effectors = trunk.node.addChild("Effectors")
        effectors.addObject("MechanicalObject", position=[[0., 0., 195.]])
        effectors.addObject('ConstraintPoint', template='Vec3', indices = [0], effectorGoal=goal_mo.position.getLinkPath(), valueType = "force", imposedValue = [0,0,0])
        effectors.addObject("BarycentricMapping", mapForces=False, mapMasses=False)

        rootNode.addObject(Controller(name="Controller", constraint_solver = constraint_solver, constraint_solver_setter=constraint_solver_setter, list_actuators = trunk.cables, list_effectors = [effectors.ConstraintPoint], list_effectors_MO = [effectors.MechanicalObject]))


    return rootNode
