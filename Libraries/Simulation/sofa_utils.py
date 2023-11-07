# -*- coding: utf-8 -*-
__authors__ = "emenager, tnavez"
__contact__ = "etienne.menager@inria.fr, tanguy.navez@inria.fr"
__version__ = "1.0.0"
__copyright__ = "(c) 2023, Inria"
__date__ = "Sept 20 2023"

import Sofa

def init_GUI(rootNode):
    """
    Init GUI in a SOFA scene.
    ----------
    Parameters
    ----------
    rootNode: Sofa.Node
        Instance of the rootNode of the considered Sofa scene
    """
    import Sofa.Gui
    Sofa.Gui.GUIManager.Init("myscene", "qglviewer")
    Sofa.Gui.GUIManager.createGUI(rootNode, __file__)
    Sofa.Gui.GUIManager.SetDimension(1080, 1080)
    Sofa.Gui.GUIManager.MainLoop(rootNode)