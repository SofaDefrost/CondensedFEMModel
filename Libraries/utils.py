# -*- coding: utf-8 -*-
__authors__ = "emenager, tnavez"
__contact__ = "etienne.menager@inria.fr, tanguy.navez@inria.fr"
__version__ = "1.0.0"
__copyright__ = "(c) 2023, Inria"
__date__ = "Sept 20 2023"

import sys
import pathlib
import time
# Local libs
import numpy as np
import torch
import torch.optim as optim

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")

import Learning.MLP.learning_tools as MLPlearning_tools
from Learning.DataSetManager import DataSetManager, get_data_loader
from Learning.normalization import compute_std_normalization, compute_minmax_normalization
from database import *
def init_network(network_name, config, batch_size=1, ratio_test_train=0.25, use_GPU=False, design_to_MM=False):
    """
    Init a neural network.

    Parameters
    ----------
    network_name: str
        The name of the network we want to initialize in [MLP].
    config: Config
        Config instance for the simulated scene
    batch_size: int
            Size of a batch of data
    ratio_test_train = float
        The number of test samples is given by ratio_test_train * n_train_sample
    use_GPU: bool
        Use GPU for matrices operations
    design_to_MM: bool
        Use network to learn (W_0, dfree_0) from design parameters.

    Outputs
    ----------
    model: model
        Encode, process and decode components
    n_samples: int
        Number of queried test samples
    dataloader: DataLoader
        Access to train data for user input config and n_samples
    dataloader_test: DataLoader
        Access to test data for user input config and n_samples
    n_constraint: int
        Number of constraints
    scaling: TODO
    optimizer: pytorch.optim
        Optimizer for the network

    model_path: str
        Path to model folder
    last_model_link: str
        Path to last registered model
    best_model_link: str
        Path to best registered model

    args: dict
        Dictionnary of model hyperparameters
    """

    N_EPOCHS = 5000  # TODO: INIT from main
    LATENT_SIZE = config.config_network["latent_size"]
    N_HIDDEN_LAYER = config.config_network["n_hidden_layers"]
    normalization_method = config.config_network["data_normalization"]
    learning_rate = config.config_network["learning_rate"]
    dropout_probability = config.config_network["dropout_probability"]

    # Use GPU if available
    if use_GPU:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

    # Assert data acquisition
    print("[INFO] >>  Init data.")
    # n_samples, sampling_strategy, id_sampling = ask_user_sampling_stat_input(config)
    n_samples = config.config_network["n_samples"]
    sampling_strategy = config.config_network["sampling_strategy"]
    id_sampling = list(query_sampling_stats(config.config_network["model_name"], sampling_strategy, n_samples))[0]["id"]
    q_train = list(
        query_sampling_stats(model_name=config.model_name, sampling_strategy=sampling_strategy, n_samples=n_samples))
    n_test_samples = int(ratio_test_train * q_train[0]['n_samples'])
    q_test = list(
        query_sampling_stats(model_name=config.model_name, sampling_strategy="Random", n_samples=n_test_samples))
    correct = True
    if q_train[0]['n_curr_sample'] < q_train[0]['n_samples'] - 1:
        print("Data Acquisition for training set is not done.")
        correct = False
    elif len(q_test) == 0:
        print("Data Acquisition for test set is not even started.")
        correct = False
    elif q_test[0]['n_curr_sample'] < q_test[0]['n_samples'] - 1:
        print("Data Acquisition for test set is not done.")
        correct = False
    if not correct:
        raise Exception("Data acquisition is not done")

    # Load data from DB
    train_dataset = DataSetManager(config.model_name, id_sampling_stats=q_train[0]['id'], train=True,
                                   design_to_MM=design_to_MM)
    test_dataset = DataSetManager(config.model_name, id_sampling_stats=q_test[0]['id'], train=False,
                                  design_to_MM=design_to_MM)
    print(">> Train set len:", train_dataset.__len__())
    print(">> Test set len:", test_dataset.__len__())

    # Init data loader
    if design_to_MM:
        n_constraint = len(train_dataset.Y[0][1])
    else:
        n_constraint = len(train_dataset.X[0][2])  # This may be changing when including collisions

    dataloader = get_data_loader(train_dataset, batch_size, True)
    dataloader_test = get_data_loader(test_dataset, batch_size, True)  # Only one batch for test set

    # Init path
    print("[INFO] >>  Init path.")
    name_design_to_MM = "design_to_MM_" if design_to_MM else ""
    use_double = "doubleMLP_" if network_name == "doubleMLP" else ""
    save_name = "MLP" if network_name == "doubleMLP" else network_name

    model_path = str(pathlib.Path(__file__).parent.absolute()) + "/../Results/Networks/" + save_name + "/"
    last_model_file = name_design_to_MM + use_double + "model_" + str(config.config_network["id"]) + "_" + str(
        id_sampling) + "_last.pth"
    last_model_link = pathlib.Path(model_path + last_model_file)
    best_model_file = name_design_to_MM + use_double + "model_" + str(config.config_network["id"]) + "_" + str(
        id_sampling) + "_best.pth"
    best_model_link = pathlib.Path(model_path + best_model_file)

    # Compute and register data normalization
    scaling = []
    if normalization_method == "Std":
        mean_features_X, std_features_X, mean_features_Y, std_features_Y = compute_std_normalization(train_dataset,
                                                                                                     design_to_MM=design_to_MM)
        scaling = [mean_features_X, std_features_X, mean_features_Y, std_features_Y]
    elif normalization_method == "MinMax":
        min_features_X, max_features_X, min_features_Y, max_features_Y = compute_minmax_normalization(train_dataset,
                                                                                                      design_to_MM=design_to_MM)
        scaling = [min_features_X, max_features_X, min_features_Y, max_features_Y]

    print("[INFO] >>  Create network.")
    if network_name == "MLP":
        if design_to_MM:
            size_W = len(train_dataset.Y[0][1])
            INPUT_SIZE = len(train_dataset.X[0])
            OUTPUT_SIZE = len(np.triu_indices(n=size_W)[0]) + size_W
        else:
            OUTPUT_SIZE = len(np.triu_indices(n=n_constraint)[0]) + n_constraint
            INPUT_SIZE = OUTPUT_SIZE + n_constraint
        model = MLPlearning_tools.MLP(INPUT_SIZE, OUTPUT_SIZE, LATENT_SIZE, n_hidden_layers=N_HIDDEN_LAYER,
                                      dropout_probability=dropout_probability)
    elif network_name == "doubleMLP":
        if design_to_MM:
            size_W = len(train_dataset.Y[0][1])
            INPUT_SIZE = len(train_dataset.X[0])
            OUTPUT_SIZE_1 = len(np.triu_indices(n=size_W)[0])
            OUTPUT_SIZE_2 = size_W
        else:
            OUTPUT_SIZE_1 = len(np.triu_indices(n=n_constraint)[0])
            OUTPUT_SIZE_2 = n_constraint
            INPUT_SIZE = OUTPUT_SIZE_1 + OUTPUT_SIZE_2 + n_constraint
        model = MLPlearning_tools.doubleMLP(INPUT_SIZE, OUTPUT_SIZE_1, OUTPUT_SIZE_2, LATENT_SIZE,
                                            n_hidden_layers=N_HIDDEN_LAYER, dropout_probability=dropout_probability)
    else:
        print("[ERROR] >> network_name not in [MLP, doubleMLP]")
        exit(1)

    # Init optimizer
    print("[INFO] >>  Init optimizer.")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Optimizer
    print(">    ... End.")

    # Init hyperparameters
    args = {}
    if config.config_network["mode_loss"] == "physics_informed":
        args.update({"lam" + str(i): float(1) for i in range(3)})
        args.update({"l" + str(i): float(1) for i in range(3)})
        args.update({"l0" + str(i): float(1) for i in range(3)})
        args["T"] = 0  # To define in main
        args['list_rho'] = (np.random.uniform(size=N_EPOCHS + 1) < 0.999).astype(int).astype(
            np.float32)  # rho probability, 0.99 by default, should be set in main
        args['rho'] = args['list_rho'][0]
        args['list_alpha'] = [1, 0] + [0.999]  # To define in main
        args["alpha"] = args['list_alpha'][0]

    return model, n_samples, id_sampling, dataloader, dataloader_test, n_constraint, scaling, optimizer, model_path, last_model_link, best_model_link, args

def ask_user_sampling_stat_input(config):
    """
    Ask user which sampling data using for training.
    ----------
    Parameters
    ----------
    config: Config
        Config instance for the simulated scene
    ----------
    Outputs
    ----------
    n_samples: int
        n_samples parameter chosen by user
    sampling_strategy: str
        Sampling strategy chosen by user
    id: int
        Unique id for chosen sampling stats.
    """
    
    print("Data are registered for model " + str(config.model_name))

    query_sampling_startegy_params = list(query_sampling_strategy_from_model(model_name=config.model_name))
    query_sampling_startegy_params = set([
        str(query_sampling_startegy_params[i]["sampling_strategy"]) for i in
        range(len(query_sampling_startegy_params))])
    print(">> The available strategies are:", query_sampling_startegy_params)
    user_input_sampling_strategy = input(
        "Which sampling strategy do you want to use? ")
    while not (user_input_sampling_strategy in query_sampling_startegy_params):
        print("Please answer a number among: ", query_sampling_startegy_params)
        user_input_sampling_strategy = input(
            "Which sampling strategy do you want to use? ")

    query_n_samples_params = list(query_n_samples_from_model(
        model_name=config.model_name, sampling_strategy=user_input_sampling_strategy))
    query_n_samples_params = [str(query_n_samples_params[i]["n_samples"])
                              for i in range(len(query_n_samples_params))]
    print(">>  For this strategy, the n_samples parameters are:",
          query_n_samples_params)
    user_input_n_samples = input("How many samples do you want to use? ")
    while not (user_input_n_samples in query_n_samples_params):
        print("Please answer a number among: " + ', '.join(query_n_samples_params))
        user_input_n_samples = input("How many samples do you want to use? ")

    query_id = list(query_id_sample_from_model(
        config.model_name, user_input_sampling_strategy, int(user_input_n_samples)))

    return int(user_input_n_samples), user_input_sampling_strategy, query_id[0]["id"]


def ask_user_sampling_properties(config_network):
    """
    Ask user which sampling data using for training.
    ----------
    Parameters
    ----------
    config_network: Dictionary
        Dictionary describing the values for each element of the considered NN. 
    ----------
    Outputs
    ----------
    data_params: list of list 
        Data simulation lines from the databse for the considered trained NN.

    """
    query_SS_properties = list(query_sampling_stats_for_a_model(config_network["model_name"]))
                
    print("Sampling strategies parameters for the model ", config_network["model_name"], ":")
    for i, params in enumerate(query_SS_properties):
        print(">> Sampling strategy n° ", i, ":", query_SS_properties[i])

    user_input_model = int(input("Which samplign strategy do you want to use ? (give the number of the SS in the previous list) "))
    while user_input_model<0 or user_input_model>i:
        print("Please answer a number in:", [0, i])
        user_input_model = int(input("Which samplign strategy do you want to use ? (give the number of the SS in the previous list) "))

    data_params = query_SS_properties[user_input_model]

    return data_params



def ask_user_n_samples_input(config):
    """
    Ask user which sampling data using for training.
    ----------
    Parameters
    ----------
    config: Config
        Config instance for the simulated scene
    ----------
    Outputs
    ----------
    user_input: int
        n_samples parameter chosen by user

    """

    connect_db()

    query_n_samples_params = list(query_n_samples_from_model(model_name=config.model_name, sampling_strategy="Grid"))
    query_n_samples_params = [str(query_n_samples_params[i]["n_samples"]) for i in range(len(query_n_samples_params))]
    print("Data are registered for model " + str(config.model_name) + " for n_samples parameters: " + ', '.join(
        query_n_samples_params))
    user_input = input("How many samples do you want to use? ")

    while not (user_input in query_n_samples_params):
        print("Please answer a number among: " + ', '.join(query_n_samples_params))
        user_input = input("How many samples do you want to use? ")

    disconnect_db()

    return int(user_input)


def ask_user_NN_properties(model_name, network_name):
    query_NN_properties = list(query_learning_stats_for_a_NN(model_name=model_name, network_name=network_name))
    print("Trained ", network_name, " parameters for the model ", model_name)
    for i, params in enumerate(query_NN_properties):
        print(">> Network n° ", i, ":", query_NN_properties[i])

    user_input_model = int(
        input("Which neural network do you want to use ? (give the number of the NN in the previous list) "))
    while user_input_model < 0 or user_input_model > i:
        print("Please answer a number in:", [0, i])
        user_input_model = int(
            input("Which neural network do you want to use ? (give the number of the NN in the previous list) "))

    params = query_NN_properties[user_input_model]

    id_sampling = params["id_sampling"]
    sampling_stats = query_sampling_stats_from_id(id_sampling)[0]

    config_network = {"id": params["id"],
                      "model_name": model_name,
                      "sampling_strategy": sampling_stats["sampling_strategy"],
                      "n_samples": sampling_stats["n_samples"],
                      "network_name": network_name,
                      "mode_loss": params["mode_loss"],
                      "n_hidden_layers": params["n_hidden_layers"],
                      "latent_size": params["latent_size"],
                      "batch_size": params["batch_size"],
                      "data_normalization": params["data_normalization"],
                      "normalization_vector": pickle.loads(params["normalization_vector"])}
    return config_network
def plot_test_loss(config):
    """
    Plot the test loss
    ----------
    Parameters
    ----------
    config: Config
        Config instance for the simulated scene
    """
    n_samples, sampling_strategy, id_sampling = ask_user_sampling_stat_input(config)
    model_path = str(pathlib.Path(__file__).parent.absolute()) + "/../Results/Networks/" + config.config_network[
        "network_name"] + "/"

    loss_by_epoch = []
    with open(pathlib.Path(
            model_path + "_model_" + str(config.config_network["id"]) + "_" + str(id_sampling) + "S_loss_test.txt"),
              "r") as f:
        lines = f.readlines()
    for line in lines:
        line_content = line.split(",")[:-1]
        loss_by_epoch.append(line_content)

    import matplotlib.pyplot as plt
    epoch = [float(loss_by_epoch[i][0]) for i in range(len(loss_by_epoch))]
    loss = [float(loss_by_epoch[i][1]) for i in range(len(loss_by_epoch))]

    # plt.scatter(epoch, loss, c ="blue",
    # 			linewidths = 2,
    # 			marker ="s",
    # 			s = 50)

    plt.plot(epoch, loss, c="blue")

    plt.xlabel("Epoch")
    plt.ylabel("Test set loss")
    plt.yscale('log')
    plt.title("Test set loss during training for " + str(n_samples) + " train samples of " + config.model_name)
    plt.show()


def create_bloc(list_matrices):
    size = list_matrices[0].shape
    block = []
    for i in range(len(list_matrices)):
        line = [np.zeros(size) for _ in range(0, i)] + [list_matrices[i]] + [np.zeros(size) for _ in range(i, len(list_matrices)-1)]
        block.append(line)
    return np.block(block)