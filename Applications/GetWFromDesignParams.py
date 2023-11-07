""" This script aims at testing predicting W from some design parameters from a learned network"""
__authors__ = "emenager, tnavez"
__contact__ = "etienne.menager@inria.fr, tanguy.navez@inria.fr"
__version__ = "1.0.0"
__copyright__ = "(c) 2022, Inria"
__date__ = "Apr 12 2023"

# System libs
import sys
import pathlib
import importlib

# Local libs
import numpy as np
import torch

# Data management libs
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../") 
learning_lib = importlib.import_module("Applications.Training")
from Libraries.database import query_learning_stats_for_a_NN, query_sampling_stats_from_id
from Libraries.utils import ask_user_NN_properties
import pickle


def W_from_design_params(config, network_name = "MLP", type_use = "comparison"):
    """
    SOFA simulation with GUI
    ----------
    Parameters
    ----------
    config: Config
        Config instance for the simulated scene
    network_name: string in [MLP]
        Name of the NN model to use. Usefull only if type_use is set to learned.
    type_use: str, in [comparison, learned, simulated]
        Specify if:
            - assessing learning results against simulation
            - obtaining the learned matrice
            - obtaining the simulated matrices
    """

    # Get W0 and dfree_0 from learned network
    if type_use == "learned" or type_use == "comparison":
        # Load network
        config.config_network = ask_user_NN_properties(config.model_name, network_name)

        # Random init of dynamical unused parameters of the learnign process
        config.config_network["learning_rate"] = 0.0001 
        config.config_network["dropout_probability"] = 0

        # Retrieve learned network
        model, _, _, _, _, n_constraint, data_scaling, _, _, _, best_model_link, _ = learning_lib.init_network(network_name, config, design_to_MM = True)

        # Get predicted W_0 and dfree_0 
        design_vars_values = [list(config.get_design_variables().values())[i][0] for i in range(len(list(config.get_design_variables().values())))]
        W_0_learned, dfree_0_learned = matrices_from_learning(network_name, model, design_vars_values,
                n_constraint, data_scaling, normalization_method = config.config_network["data_normalization"])


    # Get W0 and dfree_0 from SOFA simulation
    if type_use == "simulated" or type_use == "comparison":
        # Import libraries
        import Sofa
        import SofaRuntime
        import Sofa.Gui
        SofaRuntime.importPlugin("SofaPython3")
        from Sofa import SofaConstraintSolver
        sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../../")
        scene_lib = importlib.import_module("Models." + config.scene_name  + "." + config.model_name)
        SofaRuntime.importPlugin("SofaPython3")

        # Init scene
        root = Sofa.Core.Node("root") # Generate the root node
        config.set_action_type(is_force = True)
        scene_lib.createScene(root, config) # Create the scene graph
        Sofa.Simulation.init(root) # Initialization of the scene graph

        # Simulate one time step
        null_action = (len(config.get_actuators_variables()) + len(config.get_contacts_variables())) * [0]
        root.Controller.apply_actions(null_action)
        Sofa.Simulation.animate(root, root.dt.value)

        # Get W_0 and dfree_0
        W_0 = root.Controller.get_compliance_matrice_in_constraint_space()
        dfree_0 = root.Controller.get_dfree()


    # Print results
    if type_use == "learned" or type_use == "comparison":
        print("Learned W0:")
        print(W_0_learned)
        print("Learned dfree:")
        print(dfree_0_learned)
        print("\n")

    if type_use == "simulated" or type_use == "comparison":
        print("Simulated W0:")
        print(W_0)
        print("Simulated dfree:")
        print(dfree_0)
        print("\n")

    if type_use == "comparison":
        print("Difference W_0:")
        print(W_0 - W_0_learned)

        print("Difference dfree_0:")
        print(dfree_0 - dfree_0_learned)

        

def matrices_from_learning(network_name, model, design_params, n_constraints, scaling, normalization_method = "MinMax"):
    """
    Rebuild mechanical matrices matrices using the neural network model

    ----------
    Parameters
    ----------
    network_name: str
        The name of the neural network we use in [MLP]
    model: neural network
        The neural network we used to predict compliance and dfree value.
    design_params: list of float
        The design parameters describing the queried robot.
    n_constraint: int
        Total number of constraints on the robot
    scaling = list of list of numpy arrays
        Scaling for each component for each matrice
    normalization_method: str
        Method used for normalizing data:
            - None: No normalization method used
            - MinMax: Use minimum and maximum value for normalizing each feature
            - Std: Use mean and standard deviation for normalizing each feature
    ----------
    Outputs
    ----------
    W_0: numpy array
        Predicted compliance matrice projected in constraints space
    dfree_0: numpy array
        Predicted actuators displacement without any actuation
    """
    # Init W_0 and dfree_0 matrices
    W_0 = np.zeros((n_constraints, n_constraints))
    dfree_0 = np.zeros(n_constraints)

    if network_name == "MLP":
        MLP_lib = importlib.import_module("Learning.MLP.learning_tools")
        X = [torch.tensor(design_params)]
        Y = [torch.tensor(W_0), torch.tensor(dfree_0)]
        
        # Rescale data before providing it to the NN
        if normalization_method == "Std":
            X, _ = MLP_lib.create_data_std(X, Y, scaling[0], scaling[1], scaling[2], scaling[3], design_to_MM = True)
        elif normalization_method == "MinMax":
            X, _ = MLP_lib.create_data_minmax(X, Y, scaling[0], scaling[1], scaling[2], scaling[3], design_to_MM = True)
        else:
            X, _ = MLP_lib.create_data(X, Y, design_to_MM = True)
        
        # Prediction
        Y = model(X).detach().numpy()[0]

        dfree = Y[-n_constraints:]
        W_pred = Y[:-n_constraints]

        W = np.zeros((n_constraints, n_constraints))
        W[np.triu_indices(n=n_constraints)] = W_pred
        W[np.tril_indices(n=n_constraints, k=-1)] = W.T[np.tril_indices(n=n_constraints, k=-1)]
        W = W.reshape(-1)
    else:
        print("[ERROR] The network is not in [MLP]. Please change the network you want to use.")


    ### Compute d_free from prediction ###
    print("dfree before:", dfree)
    if normalization_method == "Std":
        dfree = dfree * (scaling[3][1]) + scaling[2][1] # Rescale dfree
    elif normalization_method == "MinMax":  
        dfree = dfree * (scaling[3][1]  - scaling[2][1]) + scaling[2][1] # Rescale dfree
    print("dfree after:", dfree)

    ### Compute W from prediction ###
    if normalization_method == "Std":
        W = W * (scaling[3][0].reshape(-1)) + scaling[2][0].reshape(-1) # Rescale W
    elif normalization_method == "MinMax":
        W = W * (scaling[3][0].reshape(-1) - scaling[2][0].reshape(-1)) + scaling[2][0].reshape(-1) # Rescale W
    W = W.reshape(n_constraints,n_constraints)
    
    return W, dfree
