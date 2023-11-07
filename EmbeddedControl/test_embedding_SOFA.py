import Sofa
import SofaRuntime
import Sofa.Gui
from Sofa import SofaConstraintSolver

import sys
import pathlib
import importlib


sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/")

SofaRuntime.importPlugin("SofaPython3")
scene_lib = importlib.import_module("PneumaticTrunk")

root = Sofa.Core.Node("root") # Generate the root node
scene_lib.createScene(root) # Create the scene graph
Sofa.Simulation.init(root) # Initialization of the scene graph

# Find out the supported GUIs
Sofa.Gui.GUIManager.Init("myscene", "qglviewer")
Sofa.Gui.GUIManager.createGUI(root, __file__)
Sofa.Gui.GUIManager.SetDimension(1080, 1080)
Sofa.Gui.GUIManager.MainLoop(root)