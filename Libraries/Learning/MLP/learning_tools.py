# -*- coding: utf-8 -*-
"""Tools for the learning (create loss, MLP).
"""

__authors__ = "emenager, tnavez"
__contact__ = "etienne.menager@inria.fr, tanguy.navez@inria.fr"
__version__ = "1.0.0"
__copyright__ = "(c) 2022, Inria"
__date__ = "Jul 13 2022"

import torch.nn as nn
import torch.nn.functional as F
import torch
import sys
import pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))

def create_loss(output_value, true_value, mode_loss, model = None, factor_reg = 0.001):
    """Create supervised loss from target and output.

    Parameters:
    -----------
        output_value: tensor
            The output of the network.
        true_value: tensor
            The true value we want to predict.
        mode_loss: string
            The name of the loss (MSE, L1, RMSE, MSEreg
    Returns:
    --------
        Loss value (tensor).
    """
    regularization = 0
    if mode_loss == "MSE": # In fact we are computing MSE Loss
        loss_function = nn.MSELoss()
    elif mode_loss == "L1":
        loss_function = nn.L1Loss()
    elif mode_loss == "RMSE":
        class RMSELoss(torch.nn.Module):
            def __init__(self):
                super(RMSELoss,self).__init__()

            def forward(self,x,y):
                eps = 0.000000000001
                criterion = nn.MSELoss()
                loss = torch.sqrt(criterion(x, y) + eps)
                return loss
        loss_function = RMSELoss()
    elif mode_loss=="MSEregL1":
        loss_function = nn.MSELoss()
        regularization = factor_reg * sum(p.abs().sum() for p in model.parameters())
    elif mode_loss == "MSEregL2":
        loss_function = nn.MSELoss()
        regularization = factor_reg * sum(p.pow(2.0).sum() for p in model.parameters())


    return loss_function(output_value, true_value) + regularization

def create_physics_informed_loss(output_value, true_value, input_value, n_constraint, n_act_constraint, normalization_method, scaling, is_contact, args:dict, factors = [100, 100]):
    """Create physics informed loss + supervised loss from target and output.

        Parameters:
        -----------
            output_value: tensor
                The output of the network.
            true_value: tensor
                The true value we want to predict.
            mode_loss: string
                The type of the supervised loss
            input_value: tensor
                The input of the network
            n_constraint: int
                The number of constraints
            n_act_constraint: int
                The number of actuation constraints
            normalization_method: string
                The normalization method used to preprocess the data
            scaling: list
                The scaling of the data. We want to scale the data for the predictive_loss, but the physics informed loss
                is true only for the unscaled data.
            is_contact: bool
                Specify if the physics loss should be computed taking contacts into account or only effectors.
            factors: list of 2 floats
                The weights for each physical term in the final loss.
                In the order (lambda_a, lambda_c).
            args: dictionary
                Dictionary of arguments describing RELOBRALO balancing scheme for the physics informed loss
        Returns:
        --------
            Loss value (tensor).
        """

    # Compute the predictive loss
    predictive_loss = create_loss(output_value, true_value, "MSE")

    batch_size = output_value.size()[0]

    #Recover dfree, W, s
    s = input_value[:, -n_constraint:]

    dfree_pred = output_value[:, -n_constraint:]

    W_pred = output_value[:, :-n_constraint]
    W = torch.zeros(size=(batch_size, n_constraint, n_constraint))
    W[:, torch.triu_indices(n_constraint, n_constraint)[0], torch.triu_indices(n_constraint, n_constraint)[1]] = W_pred
    saved_value = W[:, torch.triu_indices(n_constraint, n_constraint, offset=1)[0],
                  torch.triu_indices(n_constraint, n_constraint, offset=1)[1]]
    W.mT[:, torch.triu_indices(n_constraint, n_constraint, offset=1)[0],
    torch.triu_indices(n_constraint, n_constraint, offset=1)[1]] = saved_value

    #Rescale the elements
    if normalization_method == "Std":
        W = W * (torch.tensor(scaling[3][0])) + torch.tensor(scaling[2][0]) # Rescale W
        dfree_pred = dfree_pred * (torch.tensor(scaling[3][1])) + torch.tensor(scaling[2][1])  # Rescale dfree
        s = s * (torch.tensor(scaling[1][2])) + torch.tensor(scaling[0][2])  # Rescale s_a
    elif normalization_method == "MinMax":
        W = W * (torch.tensor(scaling[3][0]) - torch.tensor(scaling[2][0])) + torch.tensor(scaling[2][0])  # Rescale W
        dfree_pred = dfree_pred * (torch.tensor(scaling[3][1])  - torch.tensor(scaling[2][1])) + torch.tensor(scaling[2][1]) # Rescale dfree
        s = s * (torch.tensor(scaling[1][2])  - torch.tensor(scaling[0][2])) + torch.tensor(scaling[0][2]) # Rescale s_a

    if is_contact:
        #Compute the state of the constraint (delta)
        s_a = s[:, :n_act_constraint]
        s_c = s[:, n_act_constraint:]

        #Compute the dfree
        dfree_a = dfree_pred[:, :n_act_constraint]
        dfree_c = dfree_pred[:, n_act_constraint:]

        #Compute the W
        W_aa = W[:,:n_act_constraint, :n_act_constraint]
        W_ac = W[:,:n_act_constraint, n_act_constraint:]
        W_ca = W[:,n_act_constraint:, :n_act_constraint]
        W_cc = W[:,n_act_constraint:, n_act_constraint:]

        #Compute the physics inform term of the loss
        Ds_a = (s_a-dfree_a)[:, :, None]
        Ds_c = (s_c-dfree_c)[:, :, None]

        invW_aa = torch.inverse(W_aa)
        M = W_cc - torch.matmul(W_ca, torch.matmul(invW_aa, W_ac))
        invM = torch.inverse(M)
        lambda_c = torch.matmul(invM, Ds_c) - torch.matmul(invM, torch.matmul(W_ca, torch.matmul(invW_aa, Ds_a)))
        lambda_a = torch.matmul(invW_aa, Ds_a) - torch.matmul(invW_aa, torch.matmul(W_ac, lambda_c))

        physics_informed_term_a = torch.matmul(W_ac, lambda_c) + torch.matmul(W_aa, lambda_a) - Ds_a
        physics_informed_term_c = torch.matmul(W_cc, lambda_c) + torch.matmul(W_ca, lambda_a) - Ds_c

        reg_physics_informed_term_a = torch.mean(torch.norm(physics_informed_term_a.squeeze(), dim = -1))
        reg_physics_informed_term_c = torch.mean(torch.norm(physics_informed_term_c.squeeze(), dim = -1))

        # Compute RELOBRALO balancing scheme
        # adapted from https://github.com/rbischof/relative_balancing/blob/main/src/train.py
        USE_RELOBRALO_BALANCING = True

        if USE_RELOBRALO_BALANCING:
            T = float(args['T']) # Temperature coeff
            losses = [predictive_loss, reg_physics_informed_term_a, reg_physics_informed_term_c]

            # # Progress ratio for each loss
            # print("Progress ratio for each loss:")
            # print("\nPrediction:", predictive_loss / args['l00'])
            # print("\nPhysics Informed Term a:", reg_physics_informed_term_a / args['l01'])
            # print("\nPhysics Informed Term c:", reg_physics_informed_term_c / args['l02'])

            # Compute scaling based on relative improvement
            with torch.no_grad():
                lambs_hat = torch.softmax(torch.stack([losses[i]/(args['l'+str(i)]*T+1e-12) for i in range(len(losses))]), dim = 0)*len(losses)
                lambs0_hat = torch.softmax(torch.stack([losses[i]/(args['l0'+str(i)]*T+1e-12) for i in range(len(losses))]), dim = 0)*len(losses)

            # Random lookbacks: defines whether the scaling calculated in the rpevious time step of the relative improvement should be carried forward
            lambs = [args['rho']*args['alpha']*args['lam'+str(i)] + (1-args['rho'])*args['alpha']*lambs0_hat[i] + (1-args['alpha'])*lambs_hat[i] for i in range(len(losses))]

            # Final scaling obtained through exponential decay
            loss = torch.sum(torch.stack([lambs[i]*losses[i] for i in range(len(losses))]))

            # Update args
            args = args.copy()
            is_not_initialized = True
            for i in range(len(losses)):
                args['lam'+str(i)] = lambs[i]
                args['l'+str(i)] = losses[i]
                if args['l0'+str(i)] != 1:
                    is_not_initialized = False
            if is_not_initialized:
                for i in range(len(losses)):
                    args['l0'+str(i)] = losses[i]

        # No balancing scheme
        else:
            physics_informed_loss = factors[0] * reg_physics_informed_term_a  + factors[1] * reg_physics_informed_term_c
            loss = predictive_loss + physics_informed_loss

        return loss, args

    else:
        #Compute the state of the constraint (delta)
        s_a = s[:, :n_act_constraint]
        s_e = s[:, n_act_constraint:]

        #Compute the dfree
        dfree_a = dfree_pred[:, :n_act_constraint]
        dfree_e = dfree_pred[:, n_act_constraint:]

        #Compute the W
        W_aa = W[:,:n_act_constraint, :n_act_constraint]
        W_ae = W[:,:n_act_constraint, n_act_constraint:]
        W_ea = W[:,n_act_constraint:, :n_act_constraint]
        W_ee = W[:,n_act_constraint:, n_act_constraint:]

        #Compute the physics inform term of the loss
        Ds_a = (s_a-dfree_a)[:, :, None]
        Ds_e = (s_e-dfree_e)[:, :, None]

        invW_aa = torch.inverse(W_aa)
        physics_informed_term = torch.matmul(W_ea, torch.matmul(invW_aa, Ds_a)) - Ds_e

        physics_informed_loss = factors[0] * torch.mean(torch.norm(physics_informed_term.squeeze(), dim = -1))

    return predictive_loss + physics_informed_loss


class MLP(nn.Module):
    """Classical MLP.
    """
    def __init__(self, input_size, output_size, latent_size, n_hidden_layers = 1, dropout_probability = 0):
        """Initialization.

        Parameters:
        -----------
            input_size: int
                The size of the input.
            output_size: int
                The size of the output.
            latent_size: int
                The size of all the hidden layers.
            n_hidden_layers: int, default = 1
                The number of hidden layers in the MLP.
        """
        super(MLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.latent_size = latent_size
        self.n_hidden_layers = n_hidden_layers

        self.dropout_probability = dropout_probability
        self.USE_BATCH_NORM = False # Wether to use batch normalization or not

        self.network = nn.Sequential()

        self.network.add_module("input_layer", nn.Linear(self.input_size, self.latent_size))
        #self.network.add_module("input_dropout_layer", nn.Dropout(self.dropout_probability))
        self.network.add_module("input_relu_layer", nn.ReLU())
        if self.USE_BATCH_NORM:
            self.network.add_module("input_batchnorm_layer", nn.BatchNorm1d(self.latent_size))

        # Hidden layers
        for k in range(self.n_hidden_layers):
            self.network.add_module("hidden_layer_"+str(k), nn.Linear(self.latent_size, self.latent_size))
            #self.network.add_module("hidden_dropout_layer_"+str(k), nn.Dropout(self.dropout_probability))
            self.network.add_module("hidden_relu_layer_"+str(k), nn.ReLU())
            if self.USE_BATCH_NORM:
                self.network.add_module("hidden_batchnorm_layer", nn.BatchNorm1d(self.latent_size))

        # Output layer
        self.network.add_module("output_layer", nn.Linear(self.latent_size, self.output_size))

    def forward(self, x):
        """Classical forward method of nn.Module.

        Parameters:
        -----------
            x: tensor
                The input of the netwok.

        Returns:
        --------
            The output of the network.
        """
        return self.network(x)


class doubleMLP(nn.Module):
    """Learn two different and separate quantities from a same input.
    """

    def __init__(self, input_size, output_size_1, output_size_2, latent_size, n_hidden_layers=1, dropout_probability=0):
        """Initialization.

        Parameters:
        -----------
            input_size: int
                The size of the input.
            output_size1, output_size_2: int
                The size of the output_1 and the output_2.
            latent_size: int
                The size of all the hidden layers.
            n_hidden_layers: int, default = 1
                The number of hidden layers in the MLP.
        """
        super(doubleMLP, self).__init__()
        self.MLP_1 = MLP(input_size, output_size_1, latent_size, n_hidden_layers = n_hidden_layers, dropout_probability = dropout_probability)
        self.MLP_2 = MLP(input_size, output_size_2, latent_size, n_hidden_layers = n_hidden_layers, dropout_probability = dropout_probability)

    def forward(self, x):
        """Classical forward method of nn.Module.

        Parameters:
        -----------
            x: tensor
                The input of the netwok.

        Returns:
        --------
            The outputs of the network.
        """
        W = self.MLP_1(x)
        dfree = self.MLP_2(x)

        if len(x.shape) == 1:
            res = torch.cat([W, dfree], axis = 0)
        else:
            res = torch.cat([W, dfree], axis=-1)

        return res


def create_data(X_0, Y_0, design_to_MM = False):
    Y_W, Y_dfree = Y_0

    if len(Y_W.shape) == 2:
        size_W = Y_W.shape[0]
        use_batch = False

        Y_W = Y_W[np.triu_indices(n=size_W)]
        Y = torch.cat([Y_W, Y_dfree], axis=0).float()
    else:
        size_W = Y_W.shape[1]
        use_batch = True

        Y_W = Y_W[:, np.tril_indices(n=size_W)[0], np.tril_indices(n=size_W)[1]]
        Y = torch.cat([Y_W, Y_dfree], axis=-1).float()

    if design_to_MM:
        X = torch.stack(X_0).float()
    else:
        X_W_0, X_dfree_0, X_s_a = X_0
        if not use_batch:
            X_W_0 = X_W_0[np.triu_indices(n=size_W)]
            X = torch.cat([X_W_0, X_dfree_0, X_s_a]).float()
        else:
            X_W_0 = X_W_0[:, np.tril_indices(n=size_W)[0], np.tril_indices(n=size_W)[1]]
            X = torch.cat([X_W_0, X_dfree_0, X_s_a], axis=-1).float()

    return X, Y

def create_data_minmax(X_0, Y_0, min_features_X, max_features_X, min_features_Y, max_features_Y, design_to_MM = False):
    """Create scaled data to provide for training the model. Use minimum and maximum value for normalizing each feature.

    Parameters:
    -----------
        X_0: Tensor of numpy arrays
            Batch of X data 
        Y_0: Tensor of numpy arrays
            Batch of Y data
        min_features_X: list of numpy array
            Minimum values by X component evaluated on all the dataset.
        max_features_X: list of numpy array
            Maximum values by X component evaluated on all the dataset.
        min_features_Y: list of numpy array
            Minimum values by Y component evaluated on all the dataset.
        max_features_Y: list of numpy array
            Maximum values by Y component evaluated on all the dataset.
        design_to_MM: bool
        Use network to learn (W_0, dfree_0) from design parameters.

    Outputs:
    --------
        X: Tensor
            Scaled data ready for training MLP.
        Y: Tensor
            Scaled data ready for training MLP.
    """
    epsilon = 0.000000001  # For ensuring not dividing by 0

    Y_W, Y_dfree = Y_0

    if len(Y_W.shape) == 2:
        size_W = Y_W.shape[0]
        use_batch = False

        _min_features_Y = min_features_Y[0][np.triu_indices(n=size_W)]
        _max_features_Y = max_features_Y[0][np.triu_indices(n=size_W)]
        Y_W = Y_W[np.triu_indices(n=size_W)]
        Y = torch.cat([(Y_W - _min_features_Y) / (_max_features_Y - _min_features_Y + epsilon),  (Y_dfree - min_features_Y[1]) / (max_features_Y[1] - min_features_Y[1] + epsilon)], axis=0).float()
    else:
        size_W = Y_W.shape[1]
        use_batch = True

        _min_features_Y = min_features_Y[0][np.triu_indices(n=size_W)]
        _max_features_Y = max_features_Y[0][np.triu_indices(n=size_W)]
        Y_W = Y_W[:, np.triu_indices(n=size_W)[0], np.triu_indices(n=size_W)[1]]
        Y = torch.cat([(Y_W - _min_features_Y) / (_max_features_Y - _min_features_Y + epsilon),  (Y_dfree - min_features_Y[1]) / (max_features_Y[1] - min_features_Y[1] + epsilon)],  axis=-1).float()


    if design_to_MM:
        if not use_batch:
            X = ((torch.stack(X_0) - min_features_X[0]) / (max_features_X[0] - min_features_X[0] + epsilon)).float()
        else:
            X = ((torch.stack(X_0).T - min_features_X[0]) / (max_features_X[0] - min_features_X[0] + epsilon)).float()
    else:
        X_W_0, X_dfree_0, X_s_ae = X_0
        _min_features_X = min_features_X[0][np.triu_indices(n=size_W)]
        _max_features_X = max_features_X[0][np.triu_indices(n=size_W)]
        if not use_batch:
            X_W_0 = X_W_0[np.triu_indices(n=size_W)]
            X = torch.cat([(X_W_0 - _min_features_X) / (_max_features_X - _min_features_X + epsilon),  (X_dfree_0 - min_features_X[1]) / (max_features_X[1] - min_features_X[1] + epsilon), (X_s_ae - min_features_X[2]) / (max_features_X[2] - min_features_X[2] + epsilon)]).float()
        else:
            X_W_0 = X_W_0[:, np.triu_indices(n=size_W)[0], np.triu_indices(n=size_W)[1]]
            X = torch.cat([(X_W_0 - _min_features_X) / (_max_features_X - _min_features_X + epsilon), (X_dfree_0 - min_features_X[1]) / (max_features_X[1] - min_features_X[1] + epsilon),  (X_s_ae - min_features_X[2]) / (max_features_X[2] - min_features_X[2] + epsilon)],  axis=-1).float()

    return X, Y




def create_data_std(X_0, Y_0, mean_features_X, std_features_X, mean_features_Y, std_features_Y, design_to_MM = False, torch_mode = False, in_DO_loop = False):
    """Create scaled data to provide for training the model. Use mean and standard deviation for normalizing each feature.

    Parameters:
    -----------
        X_0: Tensor of numpy arrays
            Batch of X data 
        Y_0: Tensor of numpy arrays
            Batch of Y data
        mean_features_X: list of numpy array
            Mean values by X component evaluated on all the dataset.
        std_features_X: list of numpy array
            Standard deviation values by X component evaluated on all the dataset.
        mean_features_Y: list of numpy array
            Mean values by Y component evaluated on all the dataset.
        std_features_Y: list of numpy array
            Standard deviation values by Y component evaluated on all the dataset.
        design_to_MM: bool
            Use network to learn (W_0, dfree_0) from design parameters.
        torch_mode: bool
            Specify if we do the computation in torch mode or in numpy arrays.
        in_DO_loop: bool
            Specify if we are in a Design Optimizaiton loop.

    Outputs:
    --------
        X: Tensor
            Scaled data ready for training MLP.
        Y: Tensor
            Scaled data ready for training MLP.
    """
    epsilon = 0.000000001  # For ensuring not dividing by 0

    # Load data
    Y_W, Y_dfree = Y_0
    if len(Y_W.shape) == 2:
        size_W = Y_W.shape[0]
        use_batch = False

        if torch_mode == True:            
            tensor_mean_features_Y = torch.tensor(mean_features_Y[0])
            tensor_mean_features_Y = tensor_mean_features_Y[torch.triu_indices(size_W, size_W)[0], torch.triu_indices(size_W, size_W)[1]]
            tensor_std_features_Y = torch.tensor(std_features_Y[0])
            tensor_std_features_Y = tensor_std_features_Y[torch.triu_indices(size_W, size_W)[0], torch.triu_indices(size_W, size_W)[1]]
            Y_W = Y_W[torch.triu_indices(size_W, size_W)[0], torch.triu_indices(size_W, size_W)[1]]
            # print("tensor_mean_features_Y:", tensor_mean_features_Y)
            # print("tensor_std_features_Y:", tensor_std_features_Y)
            # print("Y_W after:", Y_W)
            
            Y = torch.cat([(Y_W - tensor_mean_features_Y) / (tensor_std_features_Y + epsilon), 
                           (Y_dfree - torch.tensor(mean_features_Y[1])) / (torch.tensor(std_features_Y[1]) + epsilon)], axis=0).float()
            # print("Y:", Y)
        else:
            _mean_features_Y = mean_features_Y[0][np.triu_indices(n=size_W)]
            _std_features_Y = std_features_Y[0][np.triu_indices(n=size_W)]
            Y_W = Y_W[np.triu_indices(n=size_W)]
            Y = torch.cat([(Y_W - _mean_features_Y) / (_std_features_Y + epsilon), (Y_dfree - mean_features_Y[1]) / (std_features_Y[1] + epsilon)], axis=0).float()

            
    
    else:
        if torch_mode == True: 
            # TODO: TORCH MODE
            pass
        else:
            size_W = Y_W.shape[1]
            use_batch = True

            _mean_features_Y = mean_features_Y[0][np.triu_indices(n=size_W)]
            _std_features_Y = std_features_Y[0][np.triu_indices(n=size_W)]
            Y_W = Y_W[:, np.triu_indices(n=size_W)[0], np.triu_indices(n=size_W)[1]]
            Y = torch.cat([(Y_W - _mean_features_Y) / (_std_features_Y + epsilon), (Y_dfree - mean_features_Y[1]) / (std_features_Y[1] + epsilon)], axis=-1).float()

    if design_to_MM:
        if torch_mode == True: 
            pass
        else:          
            if not use_batch:
                X = ((torch.stack(X_0) - mean_features_X[0]) / (std_features_X[0] + epsilon)).float()
            else:
                if in_DO_loop:
                    X = ((torch.stack(X_0) - mean_features_X[0]) / (std_features_X[0] + epsilon)).float()
                else:
                    X = ((torch.stack(X_0).mT - mean_features_X[0]) / (std_features_X[0] + epsilon)).float()
                
        
    else:
        X_W_0, X_dfree_0, X_s_ae = X_0
        
        if torch_mode == True:   
            tensor_mean_features_X = torch.tensor(mean_features_X[0])
            tensor_mean_features_X = tensor_mean_features_X[torch.triu_indices(size_W, size_W)[0], torch.triu_indices(size_W, size_W)[1]]
            tensor_std_features_X = torch.tensor(std_features_X[0])
            tensor_std_features_X = tensor_std_features_X[torch.triu_indices(size_W, size_W)[0], torch.triu_indices(size_W, size_W)[1]]
        
            if not use_batch:
                X_W_0 = X_W_0[torch.triu_indices(size_W, size_W)[0], torch.triu_indices(size_W, size_W)[1]]
                X = torch.cat([(X_W_0 - tensor_mean_features_X) / (tensor_std_features_X + epsilon), (X_dfree_0 - torch.tensor(mean_features_X[1])) / (torch.tensor(std_features_X[1]) + epsilon), (X_s_ae - torch.tensor(mean_features_X[2])) / (torch.tensor(std_features_X[2]) + epsilon)]).float()
            else:
                # TODO: TORCH MODE
                pass
            # print("X:", X)
        else:
            _mean_features_X = mean_features_X[0][np.triu_indices(n=size_W)]
            _std_features_X = std_features_X[0][np.triu_indices(n=size_W)]

            if not use_batch:
                X_W_0 = X_W_0[np.triu_indices(n=size_W)]
                X = torch.cat([(X_W_0 - _mean_features_X) / (_std_features_X + epsilon), (X_dfree_0 - mean_features_X[1]) / (std_features_X[1] + epsilon), (X_s_ae - mean_features_X[2]) / (std_features_X[2] + epsilon)]).float()
            else:
                X_W_0 = X_W_0[:, np.triu_indices(n=size_W)[0], np.triu_indices(n=size_W)[1]]
                X = torch.cat([(X_W_0 - _mean_features_X) / (_std_features_X + epsilon), (X_dfree_0 - mean_features_X[1]) / (std_features_X[1] + epsilon), (X_s_ae - mean_features_X[2]) / (std_features_X[2] + epsilon)], axis=-1).float()

    return X, Y
