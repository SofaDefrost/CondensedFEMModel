# -*- coding: utf-8 -*-

__authors__ = "emenager, tnavez"
__contact__ = "etienne.menager@inria.fr, tanguy.navez@inria.fr"
__version__ = "1.0.0"
__copyright__ = "(c) 2020, Inria"
__date__ = "Jun 29 2022"

# System libs
import importlib

# Data management libs
from Libraries.utils import ask_user_NN_properties
from Libraries.Simulation.sofa_utils import init_GUI


def test_simu_qp(config, network_name = "MLP", type_use = "learned", use_trajectory = True):
    """
    SOFA simulation with GUI
    ----------
    Parameters
    ----------
    config: Config
        Config instance for the simulated scene
    network_name: string in {MLP, GNN}
        Name of the learned model to use. Usefull only if from-learned is activated.
    type_use: str, in [learned, simulated, interpolated]
        To use learned matrices, compute them or interpolate them
    use_trajectory: boolean
        To use a provided trajectory for evaluating the design.
    """

    # Launch design in SOFA
    import Sofa
    import SofaRuntime
    SofaRuntime.importPlugin("SofaPython3")
    scene_lib = importlib.import_module("Models." + config.scene_name  + "." + config.model_name)

    #Ask for the network config
    if type_use == "learned":
        learning_rate = config.config_network["learning_rate"]
        dropout_probability = config.config_network["dropout_probability"]
        config.config_network = ask_user_NN_properties(config.model_name, network_name)
        config.config_network.update({"learning_rate":learning_rate, "dropout_probability": dropout_probability})

    # Generate goals to test
    goals = config.get_trajectory()

    # Launch design in SOFA
    root = Sofa.Core.Node("root") # Generate the root node
    config.set_action_type(is_force = True)
    scene_lib.createScene(root, config) # Create the scene graph

    if config.scene_name == "3BilateralFinger":
        from Libraries.Simulation.QPControllers.QPControllerContactBilateral import QPController
    elif config.scene_name in ["3ContactFinger", "ContactCubeFinger"]:
        from Libraries.Simulation.QPControllers.QPControllerContact import QPController
    # elif config.scene_name == "Diamond": # Momentaneous line for testing speed evaluation with Diamond
    #     from Libraries.Simulation.QPControllers.QPControllerSpeedAssessment import QPController
    elif config.scene_name != "2Finger":
        from Libraries.Simulation.QPControllers.QPControllerSimple import QPController
    else:
        from Libraries.Simulation.QPControllers.QPControllerCouple import QPController

    root.addObject(QPController(name="QPController", root=root, network_name=network_name, config=config, type_use=type_use, intermediate_goals=goals))
    Sofa.Simulation.init(root) # Initialization of the scene graph
    init_GUI(root)
