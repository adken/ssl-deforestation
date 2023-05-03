import os
import numpy as np
import torch
from torch.utils.data import Dataset



class TimeSeriesDataset(Dataset):
    """
    A PyTorch dataset class for loading s1 and s2 satellite time series images from a NumPy file.
    The images are expected to have dimensions N,T x C, where:
    H: height of images
    W: width of images
    T: number of timestamps in the time series
    C: number of channels in the images
    
    Parameters
    ----------
    path : str
        The path to the directory containing the sentinel1 and sentinel2 times series data.
      
    Attributes
    ----------
    s1, s2 : torch.Tensor
        Tensors containing the image data loaded from the NumPy files.
    num_samples : int
        The total number of samples (pixels) in the dataset.
    """
    
    def __init__(self, path, transforms=None):

        self.path = path
        self.dataset1 = np.load(os.path.join(self.path, 's1_stack.npy'))
        dataset2 = np.load(os.path.join(self.path, 's2_stack.npy'))
        self.x2 = torch.from_numpy(dataset2)
        self.num_samples = self.dataset1.shape[0] + self.x2.shape[0]
 

    def __len__(self):
        """
        Returns the total number of samples (pixels) in the dataset.
        """
        return self.num_samples
    
    """ def __getitem__(self, idx):
    
        Returns a single sample (pixel) from the dataset as a PyTorch tensor.
    
        s1 = self.dataset1[idx]
        m, s = np.mean(s1,axis=(0,1)), np.std(s1,axis=(0,1))

        s1 = (s1 - m) / s # Normalize s1
        x1 = torch.from_numpy(s1)
            
        return x1, self.x2[idx,:,:]/10000 
    """


    def __getitem__(self, idx):
        
        if idx < self.dataset1.shape[0]:
            s1 = self.dataset1[idx]
            m, s = np.mean(s1, axis=(0,1)), np.std(s1, axis=(0,1))
            s1 = (s1 - m) / s # Normalize s1
            x1 = torch.from_numpy(s1)
            x2 = self.x2[idx // self.x2.shape[1], idx % self.x2.shape[1], :] / 10000
        else:
            idx2 = idx - self.dataset1.shape[0]
            x1 = torch.from_numpy(self.dataset1[-1])
            x2 = self.x2[idx2 // self.x2.shape[1], idx2 % self.x2.shape[1], :] / 10000
        
        return x1, x2
