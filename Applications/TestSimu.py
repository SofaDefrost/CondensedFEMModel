# -*- coding: utf-8 -*-
"""Main file to launch script (learning, data acquisition, applications).
"""

__authors__ = "emenager, tnavez"
__contact__ = "etienne.menager@inria.fr, tanguy.navez@inria.fr"
__version__ = "1.0.0"
__copyright__ = "(c) 2020, Inria"
__date__ = "Jun 29 2022"

# System libs
import importlib

# Local libs
from Libraries.Simulation.DirectControllers.BaseVisualizationController import BaseVisualizationController
from Libraries.Simulation.sofa_utils import init_GUI


def test_simu(config, is_inverse = False, is_force = False):
    """
    SOFA simulation with GUI
    ----------
    Parameters
    ----------
    config: Config
        Config instance for the simulated scene
    is_inverse: Boolean
    	Is the scene inverse or not
    is_force: bool
        Wether actuators are controlled in force or displacement
    """
    # Launch design in SOFA
    import Sofa
    import SofaRuntime
    SofaRuntime.importPlugin("SofaPython3")
    scene_lib = importlib.import_module("Models." + config.scene_name + "." + config.model_name)
    root = Sofa.Core.Node("root") # Generate the root node
    if is_inverse:
        config.set_is_inverse()
        config.scene_config["is_force"] = True
    else:
        config.set_action_type(is_force)
        root.addObject(BaseVisualizationController(name="VisualizationController", root = root, config = config))
    scene_lib.createScene(root, config) # Create the scene graph
    Sofa.Simulation.init(root) # Initialization of the scene graph
    init_GUI(root)


