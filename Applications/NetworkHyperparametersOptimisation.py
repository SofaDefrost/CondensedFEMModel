# -*- coding: utf-8 -*-
"""Realise an optimisation of the hyper-parameters.
"""

__authors__ = "emenager, tnavez"
__contact__ = "etienne.menager@inria.fr, tanguy.navez@inria.fr"
__version__ = "1.0.0"
__copyright__ = "(c) 2022, Inria"
__date__ = "Nov 7 2022"


import sys
import os
import optuna
import pathlib
import torch
import torch.optim as optim
import numpy as np
import joblib

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))

from Libraries.utils import ask_user_sampling_stat_input, compute_std_normalization, compute_minmax_normalization
from Libraries.Learning.DataSetManager import DataSetManager, get_data_loader
from Libraries.database import *
import Libraries.Learning.MLP.learning_tools as MLPlearning_tools

PATH = str(pathlib.Path(str(pathlib.Path(__file__).parent.absolute())))
os.makedirs(PATH + "/optimisation", exist_ok=True)

def init_optimisation(config, ratio_test_train = 0.25):
    """
    Init the dataset for the optimisation.

    Parameters
    ----------
    config: Config
        Config instance for the simulated scene
    ratio_test_train: float
    	The ratio between the train set and the test set.


    Outputs
    -------
    train_dataset: DataSet 
        The train dataset, containing data for the learning step.
    test_dataset: DataSet 
        The test dataset, containing data for the validation step.  

    """
    n_samples, sampling_strategy, id_sampling = ask_user_sampling_stat_input(config)

    q_train = list(query_sampling_stats(model_name=config.model_name, sampling_strategy=sampling_strategy, n_samples=n_samples))
    n_test_samples = int(ratio_test_train * q_train[0]['n_samples']) 
    q_test = list(query_sampling_stats(model_name = config.model_name, sampling_strategy = "Random", n_samples = n_test_samples))

    train_dataset = DataSetManager(config.model_name, id_sampling_stats = q_train[0]['id'], train = True)
    test_dataset = DataSetManager(config.model_name, id_sampling_stats = q_test[0]['id'], train = False)

    return train_dataset, test_dataset

def init_network(parameters, train_dataset, test_dataset, config):
    """
    Init the network for the optimisation.

    Parameters
    ----------
    parameters: dictionary
    	The parameters we optimize (batchsize, hiddensize, hiddenlayer, learning_rate).
    train_dataset, test_dataset: DataSet
        Dataset containing data for the learning step and the validation step.
    config: Config
        Config instance for the simulated scene

    Outputs
    -------
    dataloader, dataloader_test: torch dataloaders
        Dataloader for the learning step and the validation step.
    model: torch model
        The neural network we want to optimize.
    optimizer: torch.optim optimizer
        The optimizer for the neural network.
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau
        The scheduler to change the value of the learning rate over time.
    """

    hiddensize = int(parameters["hiddensize"])
    hiddenlayer = int(parameters["hiddenlayer"])

    n_constraint = len(train_dataset.X[0][2]) # This may be changing when including collisions
    dataloader = get_data_loader(train_dataset, config.config_network["batch_size"], True)
    dataloader_test = get_data_loader(test_dataset, config.config_network["batch_size"], True) # Only one batch for test set
    
    LATENT_SIZE = hiddensize
    N_HIDDEN_LAYER = hiddenlayer

    OUTPUT_SIZE = len(np.triu_indices(n=n_constraint)[0]) + n_constraint
    INPUT_SIZE = OUTPUT_SIZE + n_constraint

    model = MLPlearning_tools.MLP(INPUT_SIZE, OUTPUT_SIZE, LATENT_SIZE, n_hidden_layers=N_HIDDEN_LAYER, dropout_probability=config.config_network["dropout_probability"])

    optimizer = optim.Adam(model.parameters(), lr=1e-3)  # Optimizer
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-05)


    return dataloader, dataloader_test, model, optimizer, scheduler


def eval_loop(model, optimizer, dataloader, scaling, config, is_train, previous_nb_update):
    """
        Test network against all dataloader set and compute loss.

        Parameters
        ----------
        model: torch model
            The neural network we want to optimize.
        optimizer: torch.optim optimizer
            The optimizer for the neural network.
        dataloader: torch dataloader
            Dataloader for the learning step/the validation step.
        scaling: list of tensor
            Scaling value for the data normalisation/standardisation
        config: Config
            Config instance for the simulated scene
        is_train: bool
            Train or test set
        previous_nb_update: int
            Number of gradient descent already done (= number of update)

        Outputs
        ----------
        e_loss: float
            Loss for evaluated model
        nb_update: int
            Number of update after the eval loop

        """

    e_loss = 0
    n_batches = 0
    nb_update = previous_nb_update


    for i, data in enumerate(dataloader):
        X_batch, Y_batch = data

        if config.config_network["data_normalization"] == "Std":
            X, Y = MLPlearning_tools.create_data_std(X_batch, Y_batch, scaling[0], scaling[1], scaling[2], scaling[3], design_to_MM=False)
        elif config.config_network["data_normalization"] == "MinMax":
            X, Y = MLPlearning_tools.create_data_minmax(X_batch, Y_batch, scaling[0], scaling[1], scaling[2],  scaling[3], design_to_MM=False)
        else:
            X, Y = MLPlearning_tools.create_data(X_batch, Y_batch, design_to_MM=False)

        Y_pred = model(X)
        if config.config_network["mode_loss"] != "physics_informed":
            loss = MLPlearning_tools.create_loss(Y_pred, Y, config.config_network["mode_loss"])
        else:
            loss = MLPlearning_tools.create_physics_informed_loss(Y_pred, Y, X, n_constraint, len(config.get_actuators_variables()),
                                    config.config_network["data_normalization"], scaling, is_contact=(len(config.get_contacts_variables()) != 0))


        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            nb_update+= 1
        
        e_loss += loss
        n_batches += 1
    e_loss /= n_batches


    
    return e_loss, nb_update


def main_loop(parameters, train_dataset, test_dataset, scaling, config):
    """
    Train the network using acquire data for a given set of parameters.

    Parameters
    ----------
    parameters: dictionary
    	The parameters we optimize (batchsize, hiddensize, hiddenlayer, learning_rate).
    train_dataset, test_dataset: DataSet
        Dataset containing data for the learning step and the validation step.
    scaling: list of tensor
            Scaling value for the data normalisation/standardisation
    config: Config
            Config instance for the simulated scene

    Outputs:
    -------
        test_loss: float
            The loss obtained on the validation set after training the network during n_epochs epochs.
    """
    dataloader, dataloader_test, model, optimizer, scheduler = init_network(parameters, train_dataset, test_dataset, config)
    nb_update, nb_update_max = 0, 1000
    while nb_update < nb_update_max:
        print(">> Number of update: {}/{}".format(nb_update, nb_update_max))
        model.train()
        e_loss, nb_update = eval_loop(model, optimizer, dataloader, scaling, config, True, nb_update)
        scheduler.step(e_loss)

    model.eval()
    with torch.no_grad():
        test_loss, nb_update = eval_loop(model, optimizer, dataloader_test, scaling, config, False, nb_update)

    return test_loss.item()
    


def objective(trial, train_dataset, test_dataset, scaling, config):
    """
    Objective function for the optimisation step.

    Parameters
    ----------
        trial: optuna trial
            The trial for the study.
        train_dataset, test_dataset: DataSet
            Dataset containing data for the learning step and the validation step.
        scaling: list of tensor
            Scaling value for the data normalisation/standardisation
        config: Config
                Config instance for the simulated scene
    
    Outputs:
    -------
        The evaluation of the network given a set of parameter.
    """
    params = { "hiddensize": trial.suggest_int("hiddensize", 32, 512, step = 1),
               "hiddenlayer": trial.suggest_int("hiddenlayer", 2, 4, step = 1)
            }

    return  main_loop(params, train_dataset, test_dataset, scaling, config)



def hyperparameters_optimisation(config, n_cores = 1, load = False, n_optimisation = 1000):
    """
    Function for the optimisation of the hyperparameters of the network. 

    Parameters:
    ----------
    config: Config
        Config instance for the simulated scene
    n_cores: int
        The number of cores for parallel optimisation. 
    load: bool
        Load the study to see results or create and run a new one.
    n_optimisation: int 
        Number of optimisation steps (corresponding to n_optimisation*20*n_cores trials).
    """
    normalization_method = config.config_network["data_normalization"]

    if not load:
        print(">>  Init dataset ...")
        train_dataset, test_dataset = init_optimisation(config)

        scaling = []
        if normalization_method == "Std":
            mean_features_X, std_features_X, mean_features_Y, std_features_Y = compute_std_normalization(train_dataset,
                                                                                                         design_to_MM=False)
            scaling = [mean_features_X, std_features_X, mean_features_Y, std_features_Y]
        elif normalization_method == "MinMax":
            min_features_X, max_features_X, min_features_Y, max_features_Y = compute_minmax_normalization(train_dataset,
                                                                                                          design_to_MM=False)
            scaling = [min_features_X, max_features_X, min_features_Y, max_features_Y]

        print(">>  Start optimisation ...")
        study = optuna.create_study(direction='minimize')
        for i in range(n_optimisation):
            study.optimize(lambda trial: objective(trial, train_dataset, test_dataset, scaling, config), n_jobs=n_cores, n_trials=20*n_cores, timeout=None)
            joblib.dump(study, PATH + "/optimisation/study.pkl")
            print(">>  Best Params ", study.best_params)


    else:
        study=joblib.load(PATH + "/optimisation/study.pkl")

        print(">>  Best trial until now:")
        print(">>    Value: ", study.best_trial.value)
        print(">>    Params: ")

        params = dict(study.best_trial.params.items())
        print("hiddensize: {},".format(params["hiddensize"]))
        print("hiddenlayer: {},".format(params["hiddenlayer"]))