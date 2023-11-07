# -*- coding: utf-8 -*-
"""CoarsePneumaticTrunk.py: create scene of the PneumaticTrunk with fine mesh
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

from AbstractPneumaticTrunk.AbstractPneumaticTrunk import PneumaticTrunkScene

def createScene(rootNode, classConfig):
    config = classConfig.get_scene_config()
    PneumaticTrunkScene(rootNode, config, nb_slices=16)


    
    
   

    
   
