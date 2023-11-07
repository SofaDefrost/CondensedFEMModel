# -*- coding: utf-8 -*-
"""Finger.py: create scene of the Finger and the controler to actuate it.
   Original code from the SoftRobot.DesignOptimization toolbox: https://github.com/SofaDefrost/SoftRobots.DesignOptimization
"""

__authors__ = "emenager, tnavez"
__contact__ = "etienne.menager@inria.fr, tanguy.navez@inria.fr"
__version__ = "1.0.0"
__copyright__ = "(c) 2020, Inria"
__date__ = "May 26 2023"

import sys
import pathlib
from math import cos, sin
import numpy as np
import os
import Sofa

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))
MeshPath = os.path.dirname(os.path.abspath(__file__))+'/Meshes/'
MeshesCachePath = os.path.dirname(os.path.abspath(__file__))+'/Meshes/Cache/'

from Generation import Cavity, Finger

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

     # For computing angle objective
    def onAnimateEndEvent(self, event):
        import math
        effector_MO = self.list_effectors_MO[0]
        CurrentPosition = np.array(effector_MO.position.value[0])
        Angle = np.abs(math.acos( abs(CurrentPosition[2]) / np.linalg.norm(CurrentPosition)))
        print("Absolute angle: ", Angle)

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


def createScene(rootNode, config):

    scene_config = config.get_scene_config()
    valueType = "force" if scene_config["is_force"] else "displacement"

    ###############################
    ### Import required plugins ###
    ###############################
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

    ##############################
    ### Visualization settings ###
    ##############################
    rootNode.addObject('LightManager')
    rootNode.addObject('PositionalLight', name="light1", color="0.8 0.8 0.8", position="0 60 50")
    rootNode.addObject('PositionalLight', name="light2", color="0.8 0.8 0.8", position="0 -60 -50")
    rootNode.addObject('VisualStyle', displayFlags='hideWireframe showBehaviorModels hideCollisionModels hideBoundingCollisionModels showForceFields showInteractionForceFields')

    ###########################
    ### Simulation settings ###
    ###########################
    rootNode.addObject('FreeMotionAnimationLoop')
    if scene_config["inverseMode"]:
        rootNode.addObject('RequiredPlugin', name='SoftRobots.Inverse')
        rootNode.addObject("QPInverseProblemSolver")
    else:
        constraint_solver = rootNode.addObject('GenericConstraintSolver', tolerance="1e-6", maxIterations="1000")
        constraint_solver_setter = rootNode.addObject("ConstraintSolverSetter")

    rootNode.findData('gravity').value = [0, 0, -9810]
    rootNode.findData('dt').value = 0.01

    solvers = rootNode.addChild('solvers')
    solvers.addObject('EulerImplicitSolver', name='odesolver', firstOrder=0, rayleighMass=0.1,  rayleighStiffness=0.1)
    solvers.addObject('SparseLDLSolver', name='precond', template = "CompressedRowSparseMatrixd")
    solvers.addObject('GenericConstraintCorrection', linearSolver="@./precond")

    ##################
    ### Load model ###
    ##################
    model = solvers.addChild('model')
    model.addObject('MeshVTKLoader', name='loader',
                    filename = config.get_mesh_filename(mode = "Volume", refine = False,
                                                        generating_function = Finger,
                                Length = config.Length, Height = config.Height, OuterRadius = config.OuterRadius,
                                TeethRadius = config.TeethRadius, PlateauHeight = config.PlateauHeight,
                                JointHeight = config.JointHeight, Thickness = config.Thickness,
                                JointSlopeAngle = config.JointSlopeAngle, FixationWidth = config.FixationWidth,
                                BellowHeight = config.BellowHeight, NBellows = config.NBellows,
                                WallThickness = config.WallThickness, CenterThickness = config.CenterThickness,
                                CavityCorkThickness = config.CavityCorkThickness, lc = config.lc_finger,
                                RefineAroundCavities=config.RefineAroundCavities))
    model.addObject('TetrahedronSetTopologyContainer', name='container', src='@loader')
    model.addObject('TetrahedronSetGeometryAlgorithms')
    model.addObject('MechanicalObject', name='tetras', template='Vec3d', showIndices='false', showIndicesScale='4e-5')
    # model.addObject('UniformMass', totalMass='0.0')
    model.addObject('TetrahedronFEMForceField', template='Vec3d', name='FEM', method='large', poissonRatio=config.PoissonRation,  youngModulus=config.YoungsModulus)

    BoxMargin = 3
    BoxCoords = [-(config.Thickness/2+BoxMargin), -BoxMargin, BoxMargin, config.Thickness/2+BoxMargin,config.Height+2*BoxMargin, -BoxMargin]
    model.addObject('BoxROI', name='boxROI', box=BoxCoords, drawBoxes=True)
    model.addObject('RestShapeSpringsForceField', points='@boxROI.indices', stiffness=1e10)

    # Visualization
    modelVisu = model.addChild('visu')
    modelVisu.addObject('MeshSTLLoader', name="loader",
                        filename = config.get_mesh_filename(mode = "Surface", refine = True,
                                                    generating_function =  Finger,
                                Length = config.Length, Height = config.Height, OuterRadius = config.OuterRadius,
                                TeethRadius = config.TeethRadius, PlateauHeight = config.PlateauHeight,
                                JointHeight = config.JointHeight, Thickness = config.Thickness, JointSlopeAngle = config.JointSlopeAngle,
                                FixationWidth = config.FixationWidth, BellowHeight = config.BellowHeight,
                                NBellows = config.NBellows, WallThickness = config.WallThickness,
                                CenterThickness = config.CenterThickness, CavityCorkThickness = config.CavityCorkThickness,
                                lc = config.lc_finger, RefineAroundCavities=config.RefineAroundCavities))
    modelVisu.addObject('OglModel', src="@loader", scale3d=[1, 1, 1])
    modelVisu.addObject('BarycentricMapping')

    # Cable Actuator
    cables = model.addChild('cables')
    cable1 = cables.addChild('cable1')

    NSegments = 3
    CableHeight = config.CableHeight
    CableHeightRelative = CableHeight-config.JointHeight
    LengthDiagonal = CableHeightRelative/np.cos(config.JointSlopeAngle)
    JointStandoff = LengthDiagonal*np.sin(config.JointSlopeAngle)

    CablePoints = np.array([])
    for i in range(NSegments):
        SegmentOffsetBase = config.Length*i
        SegmentOffsetTip  = config.Length*(i+1)
        CablePoints = np.append(CablePoints, [[0,CableHeight,-SegmentOffsetBase - JointStandoff]])
        CablePoints = np.append(CablePoints, [[0,CableHeight, -SegmentOffsetTip + JointStandoff]])

    cable1.addObject('MechanicalObject', position=CablePoints.tolist())

    if scene_config["inverseMode"]:
        cable = cable1.addObject('CableActuator', template='Vec3d', name='CableConstraint', indices=list(range(2*NSegments)), pullPoint= [0, CableHeight, 0], printLog=True)
    else:
        cable = cable1.addObject('CableConstraint', template='Vec3d', name='CableConstraint', indices=list(range(2*NSegments)), pullPoint=[0, CableHeight, 0], printLog=True, valueType = valueType)

    cable1.addObject('BarycentricMapping')

    # Goal
    goal = rootNode.addChild("Goal")
    goal_mo = goal.addObject('MechanicalObject', name='GoalMO', showObject=True, drawMode=1, showObjectScale=2,
                             showColor=[0, 1, 0, 1], position=scene_config["goalPos"])
    # Effector
    effectors = model.addChild("Effectors")
    effector_pos = [[0.0, config.Height/2, -3.0*config.Length]]
    effectors.addObject("MechanicalObject", position=effector_pos)
    effectors.addObject('ConstraintPoint', template='Vec3', indices = [i for i in range(len(effector_pos))], effectorGoal=goal_mo.position.getLinkPath(), valueType = "force", imposedValue = [0,0,0])
    effectors.addObject("BarycentricMapping", mapForces=False, mapMasses=False)

    ##################
    ### Controller ###
    ##################
    if not scene_config["inverseMode"]:
        rootNode.addObject(Controller(name="Controller", constraint_solver = constraint_solver, constraint_solver_setter=constraint_solver_setter, list_actuators = [cable], list_effectors = [effectors.ConstraintPoint], list_effectors_MO = [effectors.MechanicalObject]))

    return rootNode
