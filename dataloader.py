# Import libraries
import os
import math
import numpy as np
import pandas as pd
import glob
import rasterio
from rasterio.windows import Window
from osgeo import gdal
from rasterio import plot
from rasterio.plot import show
import ee
import xarray as xr
import rioxarray
import geemap
import geopandas as gpd
import pystac
from pystac_client import Client
import shapely.geometry
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import matplotlib.image as img
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import warnings
warnings.filterwarnings("ignore")

# Path to dataset
path = '/Users/adken/brittany/cde/Code/masters/s2.tif'

# Dataset class
class PixelDataset(Dataset):
    """
    Dataset class for satellite time series images in H, W,TXC saved as a Geotiff Image.
    H, W: height and  width of images, and HXW is which is equal to the  number of pixels in the dataset
    T: number of timestamps in the time series
    C: number of channels is the images
        """

    def __init__(self, path, window_size=1):
        self.dataset = rasterio.open(path)
        self.window_size = window_size
        
        self.height = self.dataset.height
        self.width = self.dataset.width
        self.num_bands = self.dataset.count
        self.num_samples = self.height * self.width

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        row = idx // self.width
        col = idx % self.width
        x = np.zeros((self.num_bands, self.window_size, self.window_size))
        
        # Define window centered around pixel
        win_height = min(self.window_size, self.height - row)
        win_width = min(self.window_size, self.width - col)
        row_off = max(row - self.window_size // 2, 0)
        col_off = max(col - self.window_size // 2, 0)
        window = Window(col_off, row_off, win_width, win_height)

        # Read data from the window for each band
        for i in range(self.num_bands):
            x[i,:,:] = self.dataset.read(i+1, window=window)

        return torch.from_numpy(x)


dataset = PixelDataset(path)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, prefetch_factor=2)

dataiter = iter(dataloader)
data = next(dataiter)
features = data
print(features.shape)

'The print statement returns a tensor of torch.Size([2, 130, 1, 1])'
