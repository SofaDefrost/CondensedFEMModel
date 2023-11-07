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
import os
import numpy as np

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
    goal_mo = goal.addObject('MechanicalObject', name='GoalMO', showObject = showObject, drawMode = 1, showObjectScale = 5, showColor = [237, 112, 20, 255], position=position)
    return goal_mo

class Diamond():
    """ This class is implementing a parallel soft robot.
        The robot is entirely soft and actuated with 4 cables. This robot is composed
        of a mechanical model for the deformable structure.
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

    def __init__(self, parentNode, youngModulus=3000, poissonRatio=0.45, totalMass=0.5, inverseMode=False, is_force=False):
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
            is_force: bool
                Wether actuators are controlled in force or displacement.


        """

        self.inverseMode = inverseMode
        self.is_force = is_force
        self.node = parentNode.addChild('Diamond')

        self.node.addObject('MeshVTKLoader', name="loader", filename=MeshPath+'diamond.vtk')
        self.node.addObject('MeshTopology', src="@loader")
        self.node.addObject('MechanicalObject', name="tetras", template="Vec3", showIndices=False, showIndicesScale=4e-5, rx=90, dz=35)


        self.node.addObject('UniformMass', totalMass=totalMass)
        self.node.addObject('TetrahedronFEMForceField', youngModulus=youngModulus, poissonRatio=poissonRatio)

        self.cables = self._addCables()


    def _addCables(self):
        """Private method to add cables in the Trunk.

        Outputs:
        --------
            The list of the cables.
        """
        actuators = self.node.addChild('Actuators')
        actuators.addObject('MechanicalObject', name="actuatedPoints", template="Vec3",
                    position=[[0, 0, 125], [0, 97, 45], [-97, 0, 45], [0, -97, 45], [97, 0, 45], [0, 0, 115]])

        valueType = "force" if self.is_force else "displacement"

        if self.inverseMode:
            north = actuators.addObject('CableActuator', template="Vec3", name="north" , indices=1, pullPoint=[0, 10, 30], maxPositiveDisp=20, minForce=0)
            west = actuators.addObject('CableActuator', template="Vec3", name="west", indices=2, pullPoint=[-10, 0, 30], maxPositiveDisp=20, minForce=0)
            south = actuators.addObject('CableActuator', template="Vec3", name="south", indices=3, pullPoint=[0, -10, 30], maxPositiveDisp=20, minForce=0)
            east = actuators.addObject('CableActuator', template="Vec3", name="east", indices=4, pullPoint=[10, 0, 30], maxPositiveDisp=20, minForce=0)
        else:
            north = actuators.addObject('CableConstraint', template="Vec3", name="north", indices=1, pullPoint=[0, 10, 30], valueType = valueType)
            west = actuators.addObject('CableConstraint', template="Vec3", name="west",  indices=2, pullPoint=[-10, 0, 30], valueType = valueType)
            south = actuators.addObject('CableConstraint', template="Vec3", name="south", indices=3, pullPoint=[0, -10, 30], valueType = valueType)
            east = actuators.addObject('CableConstraint', template="Vec3", name="east",  indices=4, pullPoint=[10, 0, 30], valueType = valueType)

        actuators.addObject('BarycentricMapping', mapForces=False, mapMasses=False)

        return [north, west, south, east]

    def fixExtremity(self):
        """Fix the base of the Diamond.
        """
        self.node.addObject('BoxROI', name="boxROI", box=[-15, -15, -40,  15, 15, 10], drawBoxes=True)
        self.node.addObject('FixedConstraint', indices="@boxROI.indices")

    def addEffectors(self, target, position=[[0., 0., 195.]]):
        """Add a position effector in the Trunk.
        Parameters:
        ----------
            target:
            position: list of float or list of list of float
                The position of the effector(s) in the Trunk.
        """
        effectors = self.node.addChild("Effectors")
        effectors.addObject("MechanicalObject", position=position, showObject = True, drawMode = 1, showObjectScale = 5, showColor = [0, 255, 0, 255])
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
    spotLoc = [0, 0, 500]
    rootNode.addObject("SpotLight", position=spotLoc, direction=[0, 0.0, -1])
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

    rootNode.gravity.value = [0., 0, -9810.]
    rootNode.dt.value = 0.01



    simulation = rootNode.addChild("Simulation")
    simulation.addObject('EulerImplicitSolver', name='odesolver', firstOrder= 0, rayleighMass=0.1,  rayleighStiffness=0.1)
    # simulation.addObject('ShewchukPCGLinearSolver', name='linearSolver', iterations=500, tolerance=1e-18,
    #                          preconditioners="precond")
    # simulation.addObject('SparseLDLSolver', name='precond', template = "CompressedRowSparseMatrixd")
    simulation.addObject('SparseLDLSolver', name='precond', template = "CompressedRowSparseMatrixd")
    simulation.addObject('GenericConstraintCorrection', linearSolver="@./precond")

    diamond = Diamond(simulation, inverseMode=config["inverseMode"], is_force=config["is_force"], totalMass = 0.0)
    rootNode.diamond = diamond
    diamond.fixExtremity()

    goal_mo = add_goal_node(rootNode, position = config["goalPos"], showObject = True)
    if config["inverseMode"]:
        diamond.addEffectors(goal_mo, position = [[0., 0., 125.]])
    else:
        effectors = diamond.node.addChild("Effectors")
        effectors.addObject("MechanicalObject", position=[[0., 0., 125.]])
        effectors.addObject('ConstraintPoint', template='Vec3', indices = [0], effectorGoal=goal_mo.position.getLinkPath(), valueType = "force", imposedValue = [0,0,0])
        effectors.addObject("BarycentricMapping", mapForces=False, mapMasses=False)

        rootNode.addObject(Controller(name="Controller", constraint_solver = constraint_solver,  constraint_solver_setter = constraint_solver_setter, list_actuators = diamond.cables,  list_effectors = [effectors.ConstraintPoint], list_effectors_MO = [effectors.MechanicalObject]))

    return rootNode
