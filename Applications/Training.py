# -*- coding: utf-8 -*-
"""Manage the database to interact with pytorch
"""

__authors__ = "emenager, tnavez"
__contact__ = "etienne.menager@inria.fr, tanguy.navez@inria.fr"
__version__ = "1.0.0"
__copyright__ = "(c) 2020, Inria"
__date__ = "Jul 19 2022"

# System libs
import sys
import pathlib
import time

# Local libs
import numpy as np
import torch
import torch.optim as optim
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")

import Libraries.Learning.MLP.learning_tools as MLPlearning_tools
from Libraries.utils import init_network


USE_DECAY_SCHEME = True # Use a decay scheme or not for the learnign rate


############################################################################################
################################## Network Training ########################################
############################################################################################

def train_network(network_name, config, n_train_samples = 10000, ratio_test_train = 0.25, epochs = 3000, use_GPU = False, design_to_MM = False):
    """
    Train network using acquire data.

    Parameters
    ----------
    network_name: str 
        The name of the network we want to initialize in [MLP].
    config: Config
        Config instance for the simulated scene
    n_train_samples: int
        Maximum number of train samples - Used for retrieving dataset
    ratio_test_train = float
        The number of test samples is given by ratio_test_train * n_train_sample
    epochs: int 
        The number of epochs for the training
    use_GPU: bool
        Use GPU for matrices operations
        TODO: add to(device) in graph_block and MLP (learning_tools) to make it work
    design_to_MM: bool
        Use network to learn (W_0, dfree_0) from design parameters.
    """
    batch_size = config.config_network["batch_size"]
    normalization_method = config.config_network["data_normalization"]
    model, n_samples, id_sampling, dataloader, dataloader_test, n_constraint, scaling, optimizer, model_path, last_model_link, best_model_link, args = init_network(network_name, config, batch_size, ratio_test_train, use_GPU = use_GPU, design_to_MM = design_to_MM)

    #Load precomputed network if it does exist 
    start_epoch, best_loss = 0, None 

    if pathlib.Path.exists(last_model_link):
        checkpoint = torch.load(last_model_link)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        if 'args' in checkpoint:
            args = checkpoint['args']

        print(">>   Reload the last saved model. Restart from epoch {} with train loss {}".format(start_epoch, loss))

    if pathlib.Path.exists(best_model_link):
        checkpoint = torch.load(best_model_link)
        best_epoch = checkpoint['epoch']
        test_loss = checkpoint['loss']
        best_loss = test_loss
        print(">>   Reload the best model from epoch {} with test loss {}".format(best_epoch, best_loss))

    def eval_loop(network_name, dataloader, batch_size, n_constraint, is_train, args):
        """
        Test network against all dataloader set and compute loss.
        
        Parameters
        ----------
        network_name: str 
            The name of the network we want to initialize in [MLP].
        test_dataset: DataSetManager
            Test dataset
        batch_size: int
            Size of a batch of data
        n_constraint: int
            Number of constraints
        is_train: bool
            Train or test set
        args: dict
            Dictionnary of model hyperparameters

        Outputs
        ----------
        e_loss: float
            Loss for evaluated model
        args: dict
            Updated dictionnary of model hyperparameters

        """

        e_loss, n_batches = 0, 0 
        for i, data in enumerate(dataloader):
            #print("[INFO] >> Start managing batch ", i)
            X_batch, Y_batch = data

            if network_name == "MLP" or network_name == "doubleMLP":
                if normalization_method == "Std":
                    X, Y = MLPlearning_tools.create_data_std(X_batch, Y_batch,  scaling[0], scaling[1], scaling[2], scaling[3], design_to_MM = design_to_MM)
                elif normalization_method == "MinMax":
                    X, Y = MLPlearning_tools.create_data_minmax(X_batch, Y_batch, scaling[0], scaling[1], scaling[2], scaling[3], design_to_MM = design_to_MM)
                else:
                    X, Y = MLPlearning_tools.create_data(X_batch, Y_batch, design_to_MM = design_to_MM)

                Y_pred = model(X)
                if config.config_network["mode_loss"]!= "physics_informed":
                    batch_loss = MLPlearning_tools.create_loss(Y_pred, Y, config.config_network["mode_loss"], model)
                else:
                    batch_loss, args = MLPlearning_tools.create_physics_informed_loss(Y_pred, Y,  X, n_constraint, len(config.get_actuators_variables()), normalization_method, scaling, (len(config.get_contacts_variables()) != 0), args)
            else:
                print("[ERROR]  >> The model is not in [MLP, doubleMLP]. Please change the class of the model you want to use.")

            # Backpropagation
            if is_train:
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                               
            e_loss += batch_loss
            n_batches += 1

        e_loss /= n_batches  
        return e_loss, args

    # Init scheduler for learning rate decay
    if USE_DECAY_SCHEME:
        ### Linear
        #decay_ratio = 0.99
        #lr =  (decay_ratio**start_epoch) * optimizer.param_groups[0]["lr"]

        ### Adaptative
        # The learning rate is decreased by a factor of the regression value every 10 epoches on the change value of 0.0001
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
        factor=0.1, patience=10, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-05)
    else:
        optimizer.param_groups[0]["lr"] = config.config_network["learning_rate"] # Update learning rate if needed
    
    

    # Main loop       
    print(">    Train the network ...")
    for e in range(start_epoch, epochs):

        # Test the model every X epochs
        if e % 10 == 0:
            print(">    Test the network ...")
            model.eval()
            with torch.no_grad():
                test_loss, args = eval_loop(network_name, dataloader_test, batch_size, n_constraint, False, args)
                with open(pathlib.Path(model_path + "_model_" + str(config.config_network["id"]) + "_" + str(id_sampling) + "S_loss_test.txt"), "ab") as f:
                    test_loss_data = [e, test_loss.cpu()]
                    np.savetxt(f, test_loss_data, fmt='%1.15f', newline=", ")
                    f.write(b"\n")
                print(">>   Test loss: {:.8f}".format(test_loss))
            
            # Save model if it is the best encountered 
            if best_loss == None or test_loss < best_loss:
                best_loss = test_loss
                print(">    Best model encountered so far. Saving model ...")
                torch.save({
                    'epoch': e,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': test_loss,
                    'args': args
                    }, best_model_link)
                print(">>   Model saved")
    
            print(">    ... Network tested.")
                   
        model.train()
        print(">>   Epoch {}/{}".format(e, epochs))
        start = time.time()
        e_loss, args  = eval_loop(network_name, dataloader, batch_size, n_constraint, True, args)
        if USE_DECAY_SCHEME:
            #lr *= decay_ratio
            #optimizer.param_groups[0]["lr"] = max(1e-8,lr)
            scheduler.step(e_loss)
        print("Learning rate is:", optimizer.param_groups[0]['lr'])
        end = time.time()
        print(">>   Train loss: {:.8f}".format(e_loss))
        #training_throughput = n_samples / (end - start)
        #print(f'Training throughput = {training_throughput}ms')

        # Save model   
        print(">    Saving model ...")
        torch.save({
            'epoch': e,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': e_loss,
            }, last_model_link)
        print(">>   Model saved")
                
        
        # Save deep copy of model every 30 epochs
        # if e % 30 == 0:
        #     torch.save(model, pathlib.Path(model_path + config.model_name + "_" + str(n_samples) + "S_" + str(e) +"E.pth"))
        
    print(">    ... End.")

  

############################################################################################
######################################### TOOLS ############################################
############################################################################################
