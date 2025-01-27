"""This file implements the compensation models for motion artifacts. 
Implemented models:
1- DRDNN method from Antczak, 2018"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import wfdb
import wfdb.processing
import scipy
import time
import math
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset
from sklearn.metrics import classification_report
import argparse
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from custom_dataset_for_dataloader import CustomDataset
from classification_models import NoiseDetector
from load_data_from_tensors import LoadDataFromTensor
from train import ClassificationTrainer
from plotter import Plotter


class DRDNN(nn.Module):
    def __init__(self, batch_size):
        super(DRDNN,self).__init__()

        self.lstm = nn.LSTM(input_size=120, hidden_size=120)
        self.fc1 = nn.Linear(in_features=120, out_features=120)
    """        
        self.fc1 = nn.Linear(in_features=64, out_features=64)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(in_features=64, out_features=64)
        self.relu2 = nn.ReLU()
        
        self.fc3 = nn.Linear(in_features=64, out_features=120)
    """
    def forward(self, x):

        out, (ht, ct) = self.lstm(x)
        out = out[:, -1, :]

        """        
        x = self.fc1(out)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        """
        x = self.fc1(out)

        return x
    




    
    