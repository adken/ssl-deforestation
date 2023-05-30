import os
import numpy as np
import torch
from torch.utils.data import Dataset



class TimeSeriesDataset(Dataset):    
    """
    A PyTorch dataset class for loading satellite time series images from a NumPy file.
    The images are expected to have dimensions N x T x C, where:
    N: number of samples
    T: number of timestamps in the time series
    C: number of channels in the images
    
    Parameters
    ----------
    path : str
        The path to the NumPy file containing the image data and labels.
        
    Attributes
    ----------
    x : torch.Tensor
        A tensor containing the image data loaded from the NumPy file.
    num_samples : int
        The total number of samples (pixels) in the dataset.
    """

    def __init__(self, path, transforms=None):
        self.path = path
        self.dataset1 = torch.from_numpy(np.load(os.path.join(self.path, 's1_data.npy'), mmap_mode='r'))
        self.dataset2 = torch.from_numpy(np.load(os.path.join(self.path, 's2_data.npy'), mmap_mode='r'))
        self.labels = torch.from_numpy(np.load(os.path.join(self.path, 'labels.npy')))
        self.num_samples = self.dataset1.shape[0] or self.dataset2.shape[0]
        self.transforms = transforms

    def __len__(self):
        """
        Returns the total number of samples (pixels) in the dataset.
        """
        return self.num_samples

    def __getitem__(self, idx):
        """
        Returns a single sample (pixel) from the dataset as a PyTorch tensor along with its label.
        """
        if idx < self.dataset1.shape[0]:
            s1 = self.dataset1[idx]
            s2 = self.dataset2[idx]
        else:
            idx2 = idx - self.dataset1.shape[0]
            s1 = self.dataset1[-1]
            s2 = self.dataset2[idx2]

        s1 = s1.permute(1, 0)  # swap the T and D dimensions
        s2 = s2.permute(1, 0)  # swap the T and D dimensions

        label = self.labels[idx]

        if self.transforms:
            s1 = self.transforms(s1)
            s2 = self.transforms(s2)

        return s1, s2, label

        
    
