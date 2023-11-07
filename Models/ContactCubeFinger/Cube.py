# -*- coding: utf-8 -*-
"""Create the scene with the Abstraction of Jimmy.


Units: cm, kg, s.
"""

__authors__ = ("emenager")
__contact__ = ("etienne.menager@ens-rennes.fr")
__version__ = "1.0.0"
__copyright__ = "(c) 2021, Inria"
__date__ = "August 12 2021"

import os
import numpy as np

import sys
import importlib
import pathlib

from math import cos
from math import sin

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")
meshPath = os.path.dirname(os.path.abspath(__file__))+'/Mesh/'
pathSceneFile = os.path.dirname(os.path.abspath(__file__))



def addRigidFEMObject(node, filename, collisionFilename=None, position=[0,0,0,0,0,0,1], scale=[1,1,1], textureFilename='', color=[1,1,1], density=0.002, name='Object', withSolver=True, collisionGroup = 0, withCollision=True):

    if collisionFilename == None:
        collisionFilename = filename

    object = node.addChild(name)
    object.addObject('MeshVTKLoader', name='loader', filename=collisionFilename, scale3d=scale, translation=position[:3])
    object.addObject('TetrahedronSetTopologyContainer', src='@loader', name='container')
    object.addObject('MechanicalObject', name='dofs', template='Vec3d', showIndices='false', showIndicesScale='4e-5')
    object.addObject('TetrahedronFEMForceField', template='Vec3d', name='FEM', method='large', poissonRatio=0.45,  youngModulus=1000)

    if withSolver:
        object.addObject('EulerImplicitSolver')
        object.addObject('CGLinearSolver', tolerance=1e-5, iterations=25, threshold = 1e-5)
        object.addObject('UncoupledConstraintCorrection')

    visu = object.addChild('Visu')
    visu.addObject('MeshOBJLoader', name='loader', filename=filename, scale3d=scale)
    visu.addObject('OglModel', src='@loader',  color=color if textureFilename =='' else '')
    visu.addObject('BarycentricMapping')

    CoM = object.addChild("CoM")
    CoM.addObject("MechanicalObject", template='Rigid3', position=position, showObject=True, showObjectScale=5)
    CoM.addObject("BarycentricMapping")

    return object

# Deprecated
def addRigidObject(node, filename, collisionFilename=None, position=[0,0,0,0,0,0,1], scale=[1,1,1], textureFilename='', color=[1,1,1], density=0.002, name='Object', withSolver=True, collisionGroup = 0, withCollision=True):

    if collisionFilename == None:
        collisionFilename = filename

    object = node.addChild(name)
    object.addObject('MechanicalObject', template='Rigid3', position=position, showObject=True, showObjectScale=5)

    if withSolver:
        object.addObject('EulerImplicitSolver')
        object.addObject('CGLinearSolver', tolerance=1e-5, iterations=25, threshold = 1e-5)
        object.addObject('UncoupledConstraintCorrection')

    visu = object.addChild('Visu')
    visu.addObject('MeshOBJLoader', name='loader', filename=filename, scale3d=scale)
    visu.addObject('OglModel', src='@loader',  color=color if textureFilename =='' else '')
    visu.addObject('RigidMapping')

    object.addObject('GenerateRigidMass', name='mass', density=density, src=visu.loader.getLinkPath())
    object.mass.init()
    translation = list(object.mass.centerToOrigin.value)
    #object.addObject('UniformMass', vertexMass="@mass.rigidMass")

    visu.loader.translation = translation

    if withCollision:
        collision = object.addChild('Collision')
        collision.addObject('MeshOBJLoader', name='loader', filename=collisionFilename, scale3d=scale)
        collision.addObject('MeshTopology', src='@loader')
        collision.addObject('MechanicalObject', translation=translation)
        # collision.addObject('TriangleCollisionModel', group = collisionGroup)
        # collision.addObject('LineCollisionModel', group = collisionGroup)
        # collision.addObject('PointCollisionModel', group = collisionGroup)
        collision.addObject('RigidMapping')

    return object


class Cube():
    def __init__(self, *args, **kwargs):

        if "cube_config" in kwargs:
            print(">>  Init cube_config...")
            self.name = kwargs["name"]
            self.cube_config = kwargs["cube_config"]
            self.init_pos = self.cube_config["init_pos"]
            self.density = self.cube_config["density"]
            self.scale = self.cube_config["scale"]
        else:
            print(">>  No cube_config ...")
            exit(1)

    def onEnd(self, rootNode, collisionGroup = 1):
        print(">>  Init Cube")
        self.cube = addRigidObject(rootNode,filename=meshPath+'cube.obj',name= self.name,scale=self.scale, position=self.init_pos+[0, 0, 0, 1], density=self.density, withSolver=False, collisionGroup = collisionGroup)
        #self.cube = addRigidFEMObject(rootNode,filename=meshPath+'cube.vtk',name= self.name,scale=self.scale, position=self.init_pos+[0, 0, 0, 1], density=self.density, collisionGroup = collisionGroup)

    def getPos(self):
        posCube = self.cube.MechanicalObject.position.value.tolist()
        return [posCube]

    def setPos(self, pos):
        [posCube] = pos
        self.cube.MechanicalObject.position.value = np.array(posCube)

    def addContact(self, pos_contacts):
        """Add contact point on the Cube.
        -----------
        Inputs:
        -----------
            pos_contacts: list of numpy array
                List of positions for contact points
        """
        self.contacts = []
        contacts = self.cube.Collision.addChild("Contacts")
        #contacts = self.cube.addChild("Contacts")
        for i in range(len(pos_contacts)):
            _contact = contacts.addChild("Contact_"+str(i))
            _contact.addObject("MechanicalObject", position=pos_contacts[i], showObject = True, showObjectScale = 10, showColor = "green")
            _contact.addObject('ConstraintPoint', template='Vec3', indices = [0], effectorGoal=pos_contacts[i])
            _contact.addObject("BarycentricMapping", mapForces=False, mapMasses=False, input = "@../../")
            self.contacts.append(_contact)
