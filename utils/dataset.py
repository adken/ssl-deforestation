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
        The path to the NumPy file containing the image data.
        
    Attributes
    ----------
    x : torch.Tensor
        A tensor containing the image data loaded from the NumPy file.
    num_samples : int
        The total number of samples (pixels) in the dataset.
    """
    
    def __init__(self, path, transforms=None):

        self.path = path
        self.dataset1 = np.load(os.path.join(self.path, 's1_stack.npy'), mmap_mode='r')
        self.dataset2 = np.load(os.path.join(self.path, 's2_stack.npy'), mmap_mode='r')
        self.num_samples = self.dataset1.shape[0] + self.dataset2.shape[0]
    
 
    def __len__(self):
        """
        Returns the total number of samples (pixels) in the dataset.
        """
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Returns a single sample (pixel) from the dataset as a PyTorch tensor.
        """
        if idx < self.dataset1.shape[0]:
            s1 = self.dataset1[idx]
            m, s = np.mean(s1,axis=(0,1)), np.std(s1,axis=(0,1))
            s1 = (s1 - m) / s # Normalize s1
            s1 = torch.from_numpy(s1)
            s2 = self.dataset2[idx] / 10000 # Normalize s2
            s2 = torch.from_numpy(s2)
        else:
            idx2 = idx - self.dataset1.shape[0]
            s1 = torch.from_numpy(self.dataset1[-1])
            s2 = self.dataset2[idx2] / 10000 # Normalize s2
            s2 = torch.from_numpy(s2)
        
        s1 = s1.permute(1, 0)  # swap the T and D dimensions
        s2 = s2.permute(1, 0)  # swap the T and D dimensions
        
        return s1, s2
