# -*- coding: utf-8 -*-
"""Manage the database to interact with pytorch
"""

__authors__ = "emenager, tnavez"
__contact__ = "etienne.menager@inria.fr, tanguy.navez@inria.fr"
__version__ = "1.0.0"
__copyright__ = "(c) 2020, Inria"
__date__ = "Jul 19 2022"

from torch.utils.data import Dataset, DataLoader
import torch
import pickle
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute())+"/../")
sys.path.insert(0, str(pathlib.Path(__file__).parent.absolute()))

from database import query_simulation_data

import numpy as np

def get_data_loader(dataset, batch_size, shuffle):
    """Create DataLoader frome a dataset, understandable by pytorch.

    Parameters:
    ----------
        dataset: pytorch Dataset
            The pytorch object that contains the data.
            Have two classical method: __len__, __getitem__.
        batchsize: int
            The size of the learning batch.
        shuffle: bool
            Shuffle the data in the dataset.

    Usage:
    ------
        dataset = DataSetManager("MyModel", train = True)
        print(">> Dataset len:", dataset_train.__len__())

        dataloader = get_data_loader(dataset, 32, True)

        for i, data in enumerate(dataloader):
            print(">> Start managing batch ", i)
            X_batch, Y_batch = data
            Y_predicted = f(X_batch)
            loss = norm(Y_batch - Y_predicted)

    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

class DataSetManager(Dataset):
    def __init__(self, model_name, id_sampling_stats, train = True, design_to_MM = False):
        """Manage database to create learning set.

        We try to predict Y from X: f(X) = Y.

        Parameters:
        -----------
            model_name: string
                The name of the model we want to use for the learning.
            id_sampling_stats: int
                Reference towards the corresponding sampling stats
            train: boolean (default: True)
                Use the data for the training set. Otherwise, use the
                data for the test set.
            design_to_MM: boolean (default: False)
                Use the data to train a network linking design parameters (input)
                and (W_0, dfree_0) (output).
        """
        super(DataSetManager, self).__init__()

        self.model_name = model_name
        self.id_sampling_stats = id_sampling_stats
        self.train = train
        self.design_to_MM = design_to_MM
        self.X, self.Y = [], []

        self._load()

    def _load(self):
        """Load the database and put the data into variable for the DataLoader.
        """
        print(">>   Load data ...")
        data = list(query_simulation_data(self.model_name, self.id_sampling_stats, self.train))
        
        for sample in data:
            if self.design_to_MM:
                self.X.append(pickle.loads(sample["design_params"]))
                self.Y.append([pickle.loads(sample["W_0"]), pickle.loads(sample["dfree_0"])])
            else:
                self.X.append([pickle.loads(sample["W_0"]), pickle.loads(sample["dfree_0"]), np.array(pickle.loads(sample["s_a"]) + pickle.loads(sample["s_e"]))])
                self.Y.append([pickle.loads(sample["W"]), pickle.loads(sample["dfree"])])
        print(">>   Done ...")

    def __len__(self):
        """Classical function for the DataSet object.

        Outputs:
        -------
            The size of the dataset.
        """
        return len(self.X)

    def __getitem__(self, idx):
        """Classical function for the DataSet object.

        Allows to sample one data from the dataset.

        Outputs:
        -------
            X, Y: Input and Output for the prediction f(X)=Y.
        """
        return self.X[idx], self.Y[idx]
