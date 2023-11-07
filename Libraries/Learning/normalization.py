# -*- coding: utf-8 -*-
"""Tools for the learning (create loss, MLP).
"""

__authors__ = "emenager, tnavez"
__contact__ = "etienne.menager@inria.fr, tanguy.navez@inria.fr"
__version__ = "1.0.0"
__copyright__ = "(c) 2023, Inria"
__date__ = "Oct 5 2023"


import numpy as np

def compute_minmax_normalization(train_dataset, design_to_MM=False):
    """
    Normalizing data is necessary to ensure a good gradient backpropagation in the network
    since the range of input values of the network are very different from each others.
    MinMax: Use minimum and maximum value for normalizing each feature
    ----------
    Parameters
    ----------
    train_dataset: list of tensors
        Training data loaded using DataLoader
    design_to_MM: bool
        Use normalisation on data to learn (W_0, dfree_0) from design parameters.
    ----------
    Outputs
    ----------
    min_features_X: list of numpy array
        Minimum values by X component evaluated on all the dataset.
    max_features_X: list of numpy array
        Maximum values by X component evaluated on all the dataset.
    min_features_Y: list of numpy array
        Minimum values by Y component evaluated on all the dataset.
    max_features_Y: list of numpy array
        Maximum values by Y component evaluated on all the dataset.

    """

    # Compute min/max for each input and output features
    min_W = np.minimum.reduce([train_dataset.Y[i][0] for i in range(len(train_dataset.Y))])
    max_W = np.maximum.reduce([train_dataset.Y[i][0] for i in range(len(train_dataset.Y))])

    min_dfree = np.minimum.reduce([train_dataset.Y[i][1] for i in range(len(train_dataset.Y))])
    max_dfree = np.maximum.reduce([train_dataset.Y[i][1] for i in range(len(train_dataset.Y))])

    if design_to_MM:
        min_design_param = np.minimum.reduce(train_dataset.X)
        max_design_param = np.maximum.reduce(train_dataset.X)

        return [min_design_param], [max_design_param], [min_W, min_dfree], [max_W, max_dfree]
    else:
        min_W0 = np.minimum.reduce([train_dataset.X[i][0] for i in range(len(train_dataset.X))])
        max_W0 = np.maximum.reduce([train_dataset.X[i][0] for i in range(len(train_dataset.X))])

        min_dfree0 = np.minimum.reduce([train_dataset.X[i][1] for i in range(len(train_dataset.X))])
        max_dfree0 = np.maximum.reduce([train_dataset.X[i][1] for i in range(len(train_dataset.X))])

        min_s_ae = np.minimum.reduce([train_dataset.X[i][2] for i in range(len(train_dataset.X))])
        max_s_ae = np.maximum.reduce([train_dataset.X[i][2] for i in range(len(train_dataset.X))])

        # In the current implementation, W0 and dfree_0 does not change from one training data to another
        # So it does not make sens to scale on a single value. Instead we scale X and Y together.
        min_features_Y = [np.minimum.reduce([min_W0, min_W]), np.minimum.reduce([min_dfree0, min_dfree])]
        min_features_X = min_features_Y + [min_s_ae]
        max_features_Y = [np.maximum.reduce([max_W0, max_W]), np.maximum.reduce([max_dfree0, max_dfree])]
        max_features_X = max_features_Y + [max_s_ae]
        return min_features_X, max_features_X, min_features_Y, max_features_Y

        # Later one we may use this line instead
        # return [min_W0, min_dfree0, min_s_ae], [max_W0, max_dfree0, max_s_ae], [min_W, min_dfree], [max_W, max_dfree]


def compute_std_normalization(train_dataset, design_to_MM=False):
    """
    Normalizing data is necessary to ensure a good gradient backpropagation in the network
    since the range of input values of the network are very different from each others.
    Use mean and standard deviation for normalizing each feature
    ----------
    Parameters
    ----------
    train_dataset: list of tensors
        Training data loaded using DataLoader
    design_to_MM: bool
        Use normalisation on data to learn (W_0, dfree_0) from design parameters.
    ----------
    Outputs
    ----------
    mean_features_X: list of numpy array
        Mean values by X component evaluated on all the dataset.
    std_features_X: list of numpy array
        Standard deviation values by X component evaluated on all the dataset.
    mean_features_Y: list of numpy array
        Mean values by Y component evaluated on all the dataset.
    std_features_Y: list of numpy array
        Standard deviation values by Y component evaluated on all the dataset.
    """
    # Compute mean/std for each input and output features
    # In the current implementation, W0 and dfree_0 does not change from one training data to another
    # So it does not make sens to scale on a single value. Instead we scale X and Y together.

    aggregated_W = [train_dataset.Y[i][0] for i in range(len(train_dataset.Y))]
    mean_W = np.mean(aggregated_W, axis=0)
    std_W = np.std(aggregated_W, axis=0, ddof=1)

    aggregated_dfree = [train_dataset.Y[i][1] for i in range(len(train_dataset.Y))]
    mean_dfree = np.mean(aggregated_dfree, axis=0)
    std_dfree = np.std(aggregated_dfree, axis=0, ddof=1)

    mean_features_Y = [mean_W, mean_dfree]
    std_features_Y = [std_W, std_dfree]

    if design_to_MM:
        mean_features_X = [np.mean(train_dataset.X, axis=0)]
        std_features_X = [np.std(train_dataset.X, axis=0, ddof=1)]

    else:
        aggregated_s_ae = [train_dataset.X[i][2] for i in range(len(train_dataset.X))]
        mean_s_ae = np.mean(aggregated_s_ae, axis=0)
        std_s_ae = np.std(aggregated_s_ae, axis=0, ddof=1)
        # TODO: Std for inputs W0 and dfree_0 when evolvign design
        mean_features_X = mean_features_Y + [mean_s_ae]
        std_features_X = std_features_Y + [std_s_ae]

    return mean_features_X, std_features_X, mean_features_Y, std_features_Y
