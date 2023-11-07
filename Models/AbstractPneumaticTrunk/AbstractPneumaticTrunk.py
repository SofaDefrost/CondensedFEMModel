# -*- coding: utf-8 -*-
"""AbstractPneumaticTrunk.py: create scene of the PneumaticTrunk
From the work of Paul Chaillou
"""

__authors__ = "emenager, tnavez"
__contact__ = "etienne.menager@inria.fr, tanguy.navez@inria.fr"
__version__ = "1.0.0"
__copyright__ = "(c) 2020, Inria"
__date__ = "March 23 2023"

import sys
import pathlib
import os

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))
MeshPath = os.path.dirname(os.path.abspath(__file__))+'/Mesh/'

from splib3.topology import remeshing as rf
from math import sin,cos, sqrt, acos, radians
from spicy import *
from Libraries.Simulation.BaseController import BaseController

import json
import numpy as np
from Sofa import SofaConstraintSolver


class Controller(BaseController):
    """See BaseController for a detailed description.
    """
    def __init__(self, *args, **kwargs):
        super(Controller,self).__init__(*args, **kwargs)
        self.root = kwargs["root"]
        self.actuator_state_type = kwargs["actuator_state_type"]
    def get_actuators_state(self):
        if self.actuator_state_type == "volume":
            return [float(cavity.volumeGrowth.value) for cavity in self.list_actuators]
        elif self.actuator_state_type == "pressure":
            # print("Pressures:", [float(cavity.pressure.value) for cavity in self.list_actuators])
            return [float(cavity.pressure.value) for cavity in self.list_actuators]

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


def get_cavity_box_position(r_disk, l_box):
    """Definition of the position of box cavity

    Parameters:
    ----------
        r_disk: Float
            The radius of the disk.
        l_box: Float
            The lenght of the box

    Output:
    ------
        The position of box cavity
    """
    Dx, Dy = r_disk * sin(radians(30)), r_disk * cos(radians(30))
    Ex, Ey = -r_disk, 0
    Fx, Fy = r_disk * cos(radians(60)),  - r_disk * sin(radians(60))

    D1x, D1y = Dx + l_box, Dy
    F1x, F1y = Fx + l_box, Fy
    E2x, E2y = Ex - l_box * sin(radians(30)), Ey - l_box * cos(radians(30))
    F2x, F2y = Fx - l_box * sin(radians(30)), Fy - l_box * cos(radians(30))
    D3y, D3x = Dy + l_box * cos(radians(30)), Dx - l_box * sin(radians(30))
    E3x, E3y = Ex - l_box * sin(radians(30)), Ey + l_box * cos(radians(30))

    DA3x, DA3y = (D3x + E3x)/2,  (D3y + E3y)/2
    DB3x, DB3y = (Dx + Ex)/2,  (Dy + Ey)/2
    EA2x, EA2y = (F2x + E2x)/2, (F2y + E2y)/2
    FA1x, FA1y = (F1x + D1x)/2, (F1y + D1y)/2

    III_K_x, III_K_y = Dx + 1 - 2, Dy + 0.5 - 1.5
    III_K_x2, III_K_y2 = D3x + 1 - 2, D3y + 0.5 - 1.5
    III_K_x3, III_K_y3 = DA3x + 1, DA3y + 0.5

    II_K_x, II_K_y = Ex + 1.2, Ey - 0.6
    II_K_x2, II_K_y2 = E2x + 1.2, E2y - 0.6
    II_K_x3, II_K_y3 = EA2x - 0.7, EA2y + 0.7
    II_H_x, II_H_y = EA2x + 0.7, EA2y - 0.7
    II_H_x2, II_H_y2 = F2x - 1, F2y + 1
    II_H_x3, II_H_y3 = Fx - 1, Fy + 1

    I_K_x, I_K_y = Fx, Fy + 1.5
    I_K_x2, I_K_y2 = F1x, F1y + 1.5
    I_K_x3, I_K_y3 = FA1x, FA1y - 1

    I_H_x, I_H_y = FA1x, FA1y + 1
    I_H_x2, I_H_y2 = D1x, D1y - 1.5
    I_H_x3, I_H_y3 = Dx, Dy - 1.5

    I_K = [I_K_x, I_K_y, I_K_x2, I_K_y2, I_K_x3, I_K_y3]
    I_H = [I_H_x, I_H_y, I_H_x2, I_H_y2, I_H_x3, I_H_y3]
    II_K = [II_K_x, II_K_y, II_K_x2, II_K_y2, II_K_x3, II_K_y3]
    II_H = [II_H_x, II_H_y, II_H_x2, II_H_y2, II_H_x3, II_H_y3]
    III_K = [III_K_x, III_K_y, III_K_x2, III_K_y2, III_K_x3, III_K_y3]
    DEF = [Dx, Dy, Ex, Ey, Fx, Fy]
    DA_DB_3 = [DA3x, DA3y, DB3x, DB3y, D3y, D3x, E3x, E3y]

    return [I_K, I_H, II_K, II_H, III_K, DEF, DA_DB_3]

def close_cavity(ind_top, ind_bottom):
    """Create triangles to close the cylinder mesh extremities

    Parameters:
    -----------
        ind_top: list
            Indices of the top of the cylinder
        ind_bottom: list
            Indices of the bottom of the cylinder

    Output:
    ------
        new_triangles: list
            Triangles to close the cylinder

    OUTOUT :
    new_triangles = tableau des triangles à ajouter pour fermer les cylindres
    """
    triangles = rf.close_surface(ind_top)
    triangles = rf.invers_normal(triangles)
    triangles_bottom = rf.close_surface(ind_bottom)

    for i in triangles_bottom:
        triangles.append(i)

    return triangles

def cylinder_mesh_from_ROI(points, quads, indices):
    """Create a cylinder from ROI boxes.

    """
    [new_points, ind_tab,new_quads] = rf.remesh_from_axis(points=points, mesh = quads, axis= 0, old_indices=indices)
    triangles = rf.quad_2_triangles(quads=new_quads)
    [circles, circles_ind_tab] = rf.circle_detection_regular(points = new_points, pt_per_slice = 12)
    [_, new_ind_tab_full] = rf.ordering_cylinder(circles,circles_ind_tab)
    l = len(new_ind_tab_full)

    closing_tri = close_cavity(ind_bottom = new_ind_tab_full[0],ind_top = new_ind_tab_full[l-1])
    triangles = triangles + closing_tri

    return [new_points, triangles, ind_tab]

def create_pneumatic_cavity(points, mesh, parent, num_module, num_cavity, value_type = 'pressure', init_pressure = 0, min_pressure = 0,  max_pressure = 200, inverse_mode = False):
    """Create an actuated cavity from mesh and points.

    Parameters:
    ----------
        points, mesh: list
            The list of point and triangles to create the cavity.
        parent: Sofa node
            The parent of the cavity in the Sofa hierarchy.
        num_module, num_cavity: int
            The index of the cavity, used for the name.
        value_type: string, in {pressure, volumeGrowth}, default = pressure
            Type of control of the cavity.
        init_pressure, min_pressure, max_pressure: int, default = 0, 0, 200
            Parameters to set the bounds and the initial value of the pressure.
        inverse_mode: bool
            Use the robot in a QP controler.

    Output:
    ------
        The cavity.
    """
    CavityNode = parent.addChild("Cavity" + str(num_module + 1) + str(num_cavity + 1))
    CavityNode.addObject("TriangleSetTopologyContainer", triangles=mesh, name="meshLoader", points=points)
    CavityNode.addObject('MechanicalObject', name='chambreA' + str(num_module + 1), rotation=[0, 0, 0])

    if inverse_mode:
        CavityNode.addObject('SurfacePressureActuator', name='SPC', template = 'Vec3d',triangles='@chambreAMesh'+str(num_module+1)+'.triangles', valueType=value_type, minPressure = min_pressure, maxPressure = max_pressure)
    else:
        CavityNode.addObject('SurfacePressureConstraint', name='SPC', triangles='@chambreAMesh' + str(num_module + 1) + '.triangles',
                     value=init_pressure, minPressure=min_pressure, maxPressure=max_pressure,
                     valueType=value_type)
    return CavityNode

def add_goal_node(parent, use_3_effectors, position = [0, 0, 0], showObject = True):
    """Add goal node and MO in the scene.
    Parameters:
    ----------
        parent: Sofa.Node
            The parent of the goal node in scene tree.
        use_3_effectors: bool
                Specify if we use 1 or 3 effectors for controllign the robot.
        position: list of loat or list of list of float
            The position(s) of the goal.
        showObject: bool
            Flag to indicate if we want to display the goals.
    Outputs:
    -------
        goal_mo: Sofa.MechanicalObject
            The MechanicalObject of the goal.
    """
    goal_mo = []
    goal = parent.addChild("Goal")

    if use_3_effectors:
        r, ang = 4, 2 * np.pi / 3
        position_goal = [[0, r * np.sin(i*ang+np.pi/6), r*np.cos(i*ang+np.pi/6)] for i in range(3)]

        goal.addObject("MechanicalObject", name="GoalMO", template="Rigid3d", position = position + [0, 0, 0, 1])
        for i in range(len(position_goal)):
            goal_i = goal.addChild("Goal_"+str(i))
            goal_mo_i = goal_i.addObject('MechanicalObject', name='GoalMO', showObject=showObject, drawMode=1, showObjectScale=1, showColor=[1, 0, 0, 1], position=position_goal[i])
            goal_i.addObject("RigidMapping", mapForces=False, mapMasses=False)
            goal_mo.append(goal_mo_i)

    else:
        goal_mo.append(goal.addObject('MechanicalObject', name='GoalMO', showObject=showObject, drawMode=1, showObjectScale=2, showColor=[1, 0, 0, 1], position=position))

    return goal_mo




class Stiff_Flop():
    """ This class is implementing a pneumatic trunk-like soft robot.
    The robot is entirely soft and actuated with 6 cavities.

    Parameters:
    -----------
        nb_slices: int, default = 16
            The precision of the extrusion, used to create the 3D mesh.
        init_pressure, min_pressure, max_pressure: int, default = 0, 0, 200
            Parameters to set the bounds and the initial value of the pressure.
        is_force: bool, default = True
            Use pressure or volume growth to control the robot
        inverse_mode: bool, default = False
            Use the robot in a QP controler.
    """

    def __init__(self, name = "Stiff_Flop", nb_slices = 16, init_pressure = 0, min_pressure = 0, max_pressure = 200, is_force = True, inverseMode = False):
        """
        Parameters:
        -----------
            name: string, default = "Stiff_Flop"
                The name of the robot, usefull to save points and triangle.
            nb_slices: int, default = 16
                The precision of the extrusion, used to create the 3D mesh.
            init_pressure, min_pressure, max_pressure: int, default = 0, 0, 200
                Parameters to set the bounds and the initial value of the pressure.
            is_force: bool, default = True
                Use pressure or volume growth to control the robot
            inverse_mode: bool, default = False
                Use the robot in a QP controler.
        """
        self.name = name

        self.min_pressure = min_pressure
        self.max_pressure = max_pressure
        self.init_pressure = init_pressure
        self.value_type = "pressure" if is_force else "volumeGrowth"
        self.inverse_mode = inverseMode

        self.nb_module = 2
        self.h_module = 55 #mm
        self.masse_module = 0.001#kg
        self.module_model = 'stiff_flop_indicesOK_flip.obj'
        self.modules = []

        self.YM_soft_base = 125
        self.YM_soft_top = 70
        self.YM_stiff_part = 1875
        self.coef_poi = 0.15

        self.rigid_base = 4
        self.rigid_top = 2

        self.nb_cavity = 3
        self.cavities = []

        self.r_cavity = 0.75
        self.r_disk_chamber = 4 + 1  # +1 for the box, radius of the disk, where are put the cavities
        self.r_disk_box = self.r_disk_chamber + self.r_cavity / 3  # *1.2 just to be a bit smaller
        self.l_box = self.r_cavity * 2.6  # Box length.  2* because it's radius and not diameter, and *0.6 more to be a bit bigger than the cavities itself

        self.nb_slices = nb_slices
        self.nb_beam_per_module = self.nb_slices
        self.nb_beam = self.nb_module * self.nb_beam_per_module + 1

        self.i_cavity = 0

    def create_and_init_box(self, parent, positions, module_number):
        """Create and init a box ROI.

        Parameters:
        ---------
            parent: Sofa node
                The parent of the cavity in the Sofa hierarchy.
            positions: list
                The position of the box.
            module_number: int
                The number of the considering module.
        """
        [I_K, I_H, II_K, II_H, III_K, DEF, DA_DB_3] = positions
        [Dx, Dy, Ex, Ey, Fx, Fy] = DEF
        [DA3x, DA3y, DB3x, DB3y, D3y, D3x, E3x, E3y] = DA_DB_3
        [III_K_x, III_K_y, III_K_x2, III_K_y2, III_K_x3, III_K_y3] = III_K
        [II_K_x, II_K_y, II_K_x2, II_K_y2, II_K_x3, II_K_y3] = II_K
        [II_H_x, II_H_y, II_H_x2, II_H_y2, II_H_x3, II_H_y3] = II_H
        [I_K_x, I_K_y, I_K_x2, I_K_y2, I_K_x3, I_K_y3] = I_K
        [I_H_x, I_H_y, I_H_x2, I_H_y2, I_H_x3, I_H_y3] = I_H

        display_flag = 0
        parent.addObject('BoxROI', name="DISPLAY_boxROI_III_K" + str(module_number + 1), template="Vec3d",
                         orientedBox=[III_K_x, III_K_y, self.h_module * module_number, III_K_x2, III_K_y2, self.h_module * module_number,
                                      III_K_x3, III_K_y3, self.h_module * module_number, self.h_module * 2], drawBoxes=display_flag,
                         strict=True)

        parent.addObject('BoxROI', name="DISPLAY_boxROI_III_H" + str(module_number + 1), template="Vec3d",
                         orientedBox=[DA3x - 0.6, DA3y - 0.6, self.h_module * module_number, E3x + 1.5, E3y + 0.5, 0, Ex + 1.5,
                                      Ey + 0.5, self.h_module * module_number, self.h_module * 2], drawBoxes=display_flag,
                         strict=True)

        parent.addObject('BoxROI', name="DISPLAY_boxROI_II_K" + str(module_number + 1), template="Vec3d",
                         orientedBox=[II_K_x, II_K_y, self.h_module * module_number, II_K_x2, II_K_y2, self.h_module * module_number, II_K_x3,
                                      II_K_y3, self.h_module * module_number, self.h_module * 2], drawBoxes=display_flag,
                         strict=True)

        parent.addObject('BoxROI', name="DISPLAY_boxROI_II_H" + str(module_number + 1), template="Vec3d",
                         orientedBox=[II_H_x, II_H_y, self.h_module * module_number, II_H_x2, II_H_y2, self.h_module * module_number, II_H_x3,
                                      II_H_y3, self.h_module * module_number, self.h_module * 2], drawBoxes=display_flag,
                         strict=True)

        parent.addObject('BoxROI', name="DISPLAY_boxROI_I_K" + str(module_number + 1), template="Vec3d",
                         orientedBox=[I_K_x, I_K_y, self.h_module * module_number, I_K_x2, I_K_y2, self.h_module * module_number, I_K_x3,
                                      I_K_y3, self.h_module * module_number, self.h_module * 2], drawBoxes=display_flag,
                         strict=True)

        parent.addObject('BoxROI', name="DISPLAY_boxROI_I_H" + str(module_number + 1), template="Vec3d",
                         orientedBox=[I_H_x, I_H_y, self.h_module * module_number, I_H_x2, I_H_y2, self.h_module * module_number, I_H_x3,
                                      I_H_y3, self.h_module * module_number, self.h_module * 2],
                         drawBoxes=display_flag)

        parent.addObject('BoxROI', name="boxROI_III_K" + str(module_number + 1), template="Vec3d",
                         orientedBox=[self.h_module * module_number, III_K_y, III_K_x, self.h_module * module_number, III_K_y2, III_K_x2,
                                      self.h_module * module_number, III_K_y3, III_K_x3, self.h_module * 2], drawBoxes=display_flag,
                         strict=True,
                         drawQuads=False)

        parent.addObject('BoxROI', name="boxROI_III_H" + str(module_number + 1), template="Vec3d",
                         orientedBox=[self.h_module * module_number, DA3y - 0.6, DA3x - 0.6, self.h_module * module_number, E3y + 0.5,
                                      E3x + 1.5, self.h_module * module_number, Ey + 0.5, Ex + 1.5, self.h_module * 2],
                         drawBoxes=display_flag,
                         strict=True)

        parent.addObject('BoxROI', name="boxROI_II_K" + str(module_number + 1), template="Vec3d",
                         orientedBox=[self.h_module * module_number, II_K_y, II_K_x, self.h_module * module_number, II_K_y2, II_K_x2,
                                      self.h_module * module_number, II_K_y3, II_K_x3, self.h_module * 2], drawBoxes=display_flag,
                         strict=True)

        parent.addObject('BoxROI', name="boxROI_II_H" + str(module_number + 1), template="Vec3d",
                         orientedBox=[self.h_module * module_number, II_H_y, II_H_x, self.h_module * module_number, II_H_y2, II_H_x2,
                                      self.h_module * module_number, II_H_y3, II_H_x3, self.h_module * 2], drawBoxes=display_flag,
                         strict=True)

        parent.addObject('BoxROI', name="boxROI_I_K" + str(module_number + 1), template="Vec3d",
                         orientedBox=[self.h_module * module_number, I_K_y, I_K_x, self.h_module * module_number, I_K_y2, I_K_x2,
                                      self.h_module * module_number, I_K_y3, I_K_x3, self.h_module * 2], drawBoxes=display_flag,
                         strict=True)

        parent.addObject('BoxROI', name="boxROI_I_H" + str(module_number + 1), template="Vec3d",
                         orientedBox=[self.h_module * module_number, I_H_y, I_H_x, self.h_module * module_number, I_H_y2, I_H_x2,
                                      self.h_module * module_number, I_H_y3, I_H_x3, self.h_module * 2], drawBoxes=display_flag,
                         strict=True)

        for obj in ['topo', 'engine', 'container', 'tetras']:
            node_object = parent.getObject(obj)
            node_object.init()

        object_list_box = ['boxROI_III_K' + str(module_number + 1), 'boxROI_III_H' + str(module_number + 1), 'boxROI_II_K' + str(module_number + 1),
                           'boxROI_II_H' + str(module_number + 1), 'boxROI_I_K' + str(module_number + 1), 'boxROI_I_H' + str(module_number + 1)]
        for obj in object_list_box:
            node_object = parent.getObject(obj)
            node_object.init()

        return object_list_box

    def create_module_extrude(self, parent, num_module, module_model_path):
        """Function to create one module by extruding the base.

        Parameters:
        ---------
            parent: Sofa node
                The parent of the cavity in the Sofa hierarchy.
            num_module: int
                The number of the module.
            module_model_path: string
                The path to the base to extrude.

        """
        module = parent.addChild('stiff_flop' + str(num_module + 1))
        object_list = ['topo', 'engine', 'container', 'modifier', 'tetras']
        module.addObject('MeshOBJLoader', name=object_list[0], filename=module_model_path, translation=[self.h_module * num_module, 0, 0], rotation=[0, 0, 90])
        engine = module.addObject('ExtrudeQuadsAndGenerateHexas', name=object_list[1], template='Vec3d',
                                  thicknessIn='0.0', thicknessOut=-self.h_module, numberOfSlices=self.nb_slices,
                                  surfaceVertices='@topo.position', surfaceQuads='@topo.quads')

        engine.init()
        hexas = engine.extrudedHexas.value
        points = engine.extrudedVertices.value

        [new_points, new_points_l, new_hexas] = rf.remesh_from_axis(points=points, mesh=hexas, axis=2)

        engine.extrudedHexas.value = new_hexas
        engine.extrudedVertices.value = new_points

        module.addObject('HexahedronSetTopologyContainer', position='@engine.extrudedVertices',
                         hexas='@engine.extrudedHexas', name=object_list[2])
        module.addObject('MechanicalObject', name=object_list[4], template="Vec3d", position='@container.position',
                         showIndices="false", showIndicesScale="4e-5", ry="0",
                         rz="0")
        module.addObject('UniformMass', totalMass=self.masse_module)

        return module

    def create_cavity(self, module, num_module):
        """Function to create the cavity.

        Parameters:
        ---------
            module: Sofa object
                The considering module.
            num_module: int
                The number of the module.

        """
        self.create_and_init_box(parent=module, positions= get_cavity_box_position(r_disk=self.r_disk_box, l_box=self.l_box), module_number=num_module)
        file = MeshPath + self.name + "_module_"+str(num_module) + ".txt"
        if os.path.isfile(file):
            with open(file) as f:
                data = json.load(f)
        else:
            data = [{} for _ in range(self.nb_cavity)]


        for j in range(self.nb_cavity):
            if os.path.isfile(file):
                new_points2 = data[j]["points"]
                triangles = data[j]["triangles"]
            else:

                name_base = "boxROI_" + "I" * (j + 1)
                Boite_K = module.getObject(name_base + "_K" + str(num_module + 1))
                Boite_H = module.getObject(name_base + "_H" + str(num_module + 1))

                [new_points_K, triangles_K, _] = cylinder_mesh_from_ROI(points=Boite_K.pointsInROI.value, quads=Boite_K.quadInROI.value, indices=Boite_K.indices.value)
                [new_points_H, triangles_H, _] = cylinder_mesh_from_ROI(points=Boite_H.pointsInROI.value, quads=Boite_H.quadInROI.value, indices=Boite_H.indices.value)

                nb_tri = len(new_points_K)
                for d in range(len(triangles_H)):
                    triangles_H[d] = [triangles_H[d][0] + nb_tri, triangles_H[d][1] + nb_tri, triangles_H[d][2] + nb_tri]

                triangles = [*triangles_K, *triangles_H]
                new_points = [*new_points_K, *new_points_H]

                [new_points2, _, triangles] = rf.remesh_from_axis(points=new_points, mesh=triangles,  axis=2)
                triangles = rf.invers_normal(triangles)

                data[j]["points"] = [p.tolist() for p in new_points2]
                data[j]["triangles"]=[[int(t_i) for t_i in t] for t in triangles]

            CavityNode = create_pneumatic_cavity(points=new_points2, mesh=triangles, parent=module, num_module=num_module, num_cavity=j,  value_type=self.value_type, init_pressure = self.init_pressure, min_pressure = self.min_pressure,  max_pressure = self.max_pressure, inverse_mode = self.inverse_mode)
            self.cavities.append(CavityNode.SPC)

            self.i_cavity = self.i_cavity + 1
            if self.i_cavity == self.nb_cavity:
                self.i_cavity = 0

            CavityNode.addObject('AdaptiveBeamMapping', interpolation='@../../BeamInterpolation', input='@../../DOFs',  output='@./chambreA' + str(num_module + 1))  # classic

        if not os.path.isfile(file):
            with open(file, 'w') as f:
                json.dump(data, f)


    def createRobot(self, parent):
        """Function to create the robot.

        Parameters:
        ---------
            parent: Sofa node
                The parent of the cavity in the Sofa hierarchy.

        """
        for num_module in range(self.nb_module):

            module_model_path = MeshPath + self.module_model
            module = self.create_module_extrude(parent, num_module, module_model_path)

            if num_module == 0:
                module.addObject('HexahedronFEMForceField', template='Vec3d', name='FEM', method='large',
                                 poissonRatio=self.coef_poi, youngModulus=self.YM_soft_base)
            else:
                module.addObject('HexahedronFEMForceField', template='Vec3d', name='FEM', method='large',
                                 poissonRatio=self.coef_poi, youngModulus=self.YM_soft_top)

            module.addObject('AdaptiveBeamMapping', interpolation='@../BeamInterpolation', input='@../DOFs',
                             output='@./tetras')


            self.create_cavity(module=module, num_module=num_module)
            self.modules.append(module)


    def addEffectors(self, target, use_3_effectors, showObject = True):
        """Add a position effector in the PneumaticTrunk.

        Warning: the robot is rotate by 90°.

        Parameters:
        ----------
            use_3_effectors: bool
                Specify if we use 1 or 3 effectors for controllign the robot.
            showObject: bool
                If we want to see the effector or not
        """

        position = [self.nb_module*self.h_module, 0, 0]

        if use_3_effectors:
            r, ang = 4, 2*np.pi/3
            position_goal = [[position[0], position[1] + r * np.sin(i * ang+np.pi/6), position[2] + r * np.cos(i * ang+np.pi/6)] for i in
                            range(3)]

        effector_MO, effector_constraint = [], []
        effectors = self.modules[-1].addChild("Effectors")

        if use_3_effectors:
            for i in range(3):
                effectors_i = effectors.addChild("Effectors_"+str(i))
                effectors_i_MO = effectors_i.addObject("MechanicalObject", position= position_goal[i], showObject=showObject, drawMode=1, showObjectScale=1, showColor=[0, 1, 0, 1])
                effector_i_constraint = effectors_i.addObject('ConstraintPoint', template='Vec3', indices = [i for i in range(len(position_goal[i]))], effectorGoal= target[i].position.getLinkPath(), valueType = "force", imposedValue = [0,0,0])
                effectors_i.addObject("BarycentricMapping", mapForces=False, mapMasses=False, input = "@../../")

                effector_MO.append(effectors_i_MO)
                effector_constraint.append(effector_i_constraint)

        else:
            effectors_i = effectors.addChild("Effectors_"+str(0))
            effectors_i_MO = effectors_i.addObject("MechanicalObject", position= [position], showObject=showObject, drawMode=1, showObjectScale=1, showColor=[0, 1, 0, 1])
            effector_i_constraint = effectors_i.addObject('ConstraintPoint', template='Vec3', indices = [0], effectorGoal= target[0].position.getLinkPath(), valueType = "force", imposedValue = [0,0,0])
            effectors_i.addObject("BarycentricMapping", mapForces=False, mapMasses=False)

            effector_MO.append(effectors_i_MO)
            effector_constraint.append(effector_i_constraint)

        return effector_MO, effector_constraint


def PneumaticTrunkScene(rootNode, config, nb_slices = 16, dt = 0.1):
    if config["inverseMode"]:
        print("[ERROR] >> This scene is not design to work in inverse mode.")
        exit(1)

    PneumaticTrunk = Stiff_Flop(name = config["name"], nb_slices = nb_slices, is_force = config["is_force"])
    rootNode.addObject('RequiredPlugin', name='BeamAdapter')
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
                                                 "Sofa.Component.Engine.Generate"
                                             ])

    rootNode.findData('gravity').value=[0, 0, 0]
    rootNode.addObject('VisualStyle', displayFlags='showVisualModels showBehaviorModels showCollisionModels hideBoundingCollisionModels showForceFields showInteractionForceFields hideWireframe')
    rootNode.addObject('BackgroundSetting', color='0 0.168627 0.211765')
    rootNode.findData('dt').value= dt
    constraint_solver = rootNode.addObject('GenericConstraintSolver', maxIterations='100', tolerance='0.0000001')
    constraint_solver_setter = rootNode.addObject("ConstraintSolverSetter")

    rootNode.addObject('FreeMotionAnimationLoop')
    rootNode.addObject('DefaultVisualManagerLoop')

    rigidFramesNode  = rootNode.addChild('RigidFrames')
    rigidFramesNode.addObject('EulerImplicitSolver', firstOrder='0', vdamping=0, rayleighStiffness='0.3',rayleighMass='0.1')
    rigidFramesNode.addObject('SparseLDLSolver', name='ldlsolveur',template="CompressedRowSparseMatrixd")
    rigidFramesNode.addObject('GenericConstraintCorrection')
    rigidFramesNode.addObject('RegularGridTopology',  name='meshLinesCombined',  nx=PneumaticTrunk.nb_beam, ny='1', nz='1', xmax=PneumaticTrunk.h_module*PneumaticTrunk.nb_module, xmin='0.0', ymin='0', ymax='0',zmin='0',zmax='0')
    rigidFramesNode.addObject('MechanicalObject',  name='DOFs', template='Rigid3d', showObject='1', showObjectScale='1', rotation=[0, 0 ,0], translation = [0,0,0]) # -90 on y
    rigidFramesNode.addObject('BeamInterpolation', name='BeamInterpolation', printLog = '0',  dofsAndBeamsAligned='true', straight='1', crossSectionShape='circular', radius=PneumaticTrunk.r_cavity + PneumaticTrunk.r_disk_chamber)
    rigidFramesNode.addObject('RestShapeSpringsForceField', name='anchor', points='0', stiffness='1e12', angularStiffness='1e12')

    goal_mo = add_goal_node(rootNode, use_3_effectors = config["use3Effectors"], position=config["rigidGoalPos"][:3], showObject=True)
    PneumaticTrunk.createRobot(parent = rigidFramesNode)
    effector_MO, effector_constraint = PneumaticTrunk.addEffectors(target = goal_mo, use_3_effectors = config["use3Effectors"])

    rootNode.addObject(Controller(name="Controller", constraint_solver=constraint_solver, constraint_solver_setter = constraint_solver_setter,
                                    list_actuators=PneumaticTrunk.cavities,
                                  list_effectors=effector_constraint,
                                  list_effectors_MO=effector_MO, root = rootNode,
                                  actuator_state_type = config["actuatorStateType"]))

    return rootNode
