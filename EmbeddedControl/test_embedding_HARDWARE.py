import sys
import pathlib
import importlib


sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/")
controler_lib = importlib.import_module("HardwareController")

controller = controler_lib.EmbeddedController()
controller.simulate()