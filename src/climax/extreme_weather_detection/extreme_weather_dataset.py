from torch.utils.data import Dataset
import os
import numpy as np
import torch

class ExtremeWeatherDataset(Dataset):
    """
    Pytorch Dataset class for the ClimateNet dataset
    """

    def __init__(self, root_dir, transform=None, target_transform=None):
        """
        Constructor for the ExtreneWeatherDataset class

        Arguments:
            # TODO: decide the structure of the directory
            root_dir: string - a path to a directory containing a data and labels npy array
            transform: function - a function that will be appied to all input samples before they are returned to the user
            target_transform: function - a function that will be applied to all labels before they are returned to the user

        Returns: None
        """
        
        self.data = torch.from_numpy(np.load(os.path.join(root_dir, "data.npy")))
        self.labels = torch.from_numpy(np.load(os.path.join(root_dir, "labels.npy")))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        """
        Gets the length of the dataset by using the length of the labels
        
        Arguments:
        
        Returns:
            int - the length of the dataset
        """
        return len(self.labels)
    
    def __getitem__(self, i):
        """
        Gets a (data, label) pair from the dataset using the specified index, and applies the appropriate transforms to them
        
        Arguments:
            i: int - the index of the dataset we will retrieve

        Returns:
            tuple: (data, label) pair representing the tranformed data and label at index i of the dataset
        """
        # retrieve data sample and label using index
        data_i = self.data[i]
        label_i = self.labels[i]

        # apply appropriate transforms
        if self.transform is not None:
            data_i = self.transform(data_i)
        if self.target_transform is not None:
            label_i = self.target_transform(label_i)

        # return data, label pair
        return data_i, label_i
