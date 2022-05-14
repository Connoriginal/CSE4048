import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.io as sio

class MNIST3D(Dataset):
    def __init__(self, train = True, transform = None, target_transform = None):
        if train :
            self.df = pd.read_csv('./data/MNIST3d/train.csv')
        else :
            self.df = pd.read_csv('./data/MNIST3d/test.csv')
        self.transform = transform
        self.target_transform = target_transform
        
        df_images = self.df.loc[:, self.df.columns != 'label']
        df_labels = self.df['label']
        
        self.images = df_images.values.reshape(-1, 28, 28, 3).astype(np.float32)
        self.labels = df_labels.values
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

class DigitData(Dataset):
    def __init__(self, train = True, transform = None, target_transform = None):
        if train :
            self.df = pd.read_csv('./data/DigitData/train.csv')
        else :
            self.df = pd.read_csv('./data/DigitData/test.csv')
        self.transform = transform
        self.target_transform = target_transform
        
        df_images = self.df.loc[:, self.df.columns != 'label']
        df_labels = self.df['label']
        
        self.images = df_images.values.reshape(-1, 28, 28, 3).astype(np.float32)
        self.labels = df_labels.values
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

class SVHN(Dataset):
    def __init__(self, train = True, transform = None, target_transform = None):
        if train :
            loaded_mat = sio.loadmat('./data/SVHN/train_32x32.mat')
        else :
            loaded_mat = sio.loadmat('./data/SVHN/test_32x32.mat')
        
        self.data = loaded_mat['X']
        self.labels = loaded_mat['y'].astype(np.int64).squeeze()
        np.place(self.labels, self.labels == 10, 0)
        self.data = np.transpose(self.data, (3, 0, 1, 2))
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.data[idx]
        image = np.resize(image,(28,28,3)) 
        label = int(self.labels[idx])

        
        if self.transform:
            image = self.transform(image) * 255
            
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
        


class TotalDataset(Dataset):
    def __init__(self, train = True, transform = None, target_transform = None):
        if train :
            self.df = pd.read_csv('./data/MNIST3d/train.csv')
            self.df2 = pd.read_csv('./data/DigitData/train.csv')
        else :
            self.df = pd.read_csv('./data/MNIST3d/test.csv')
            self.df2 = pd.read_csv('./data/DigitData/test.csv')
        self.transform = transform
        self.target_transform = target_transform

        df_images = self.df.loc[:, self.df.columns != 'label']
        df_labels = self.df['label']

        df_images2 = self.df2.loc[:, self.df2.columns != 'label']
        df_labels2 = self.df2['label']

        self.images = np.concatenate((df_images.values.reshape(-1, 28, 28, 3).astype(np.float32), df_images2.values.reshape(-1, 28, 28, 3).astype(np.float32)), axis = 0)
        self.labels = np.concatenate((df_labels.values, df_labels2.values), axis = 0)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
        