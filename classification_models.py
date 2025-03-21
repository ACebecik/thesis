"""This file implements the classification models for the quality assessment task.
Implemented models:
1- CNN from Ansari, 2018"""

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



class NoiseDetector(nn.Module):
    def __init__(self, in_channels=1, p_dropout=0.3, fc_size = 1024):
        super(NoiseDetector,self).__init__()

        # First set of Conv,Relu,Pooling,Dropout
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=7, stride=1, padding=3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout(p=p_dropout)

        # 2nd
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=7, stride=1, padding=3)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout(p=p_dropout)

        #3rd
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.drop3 = nn.Dropout(p=p_dropout)

        #4th
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=7, stride=1, padding=3)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.drop4 = nn.Dropout(p=p_dropout)

        #FC layers
        self.flatten1 = nn.Flatten()
        self.fc1 = nn.Linear(in_features=448, out_features=fc_size)
        self.relu5 = nn.ReLU()

        self.fc2 = nn.Linear(in_features=fc_size, out_features=1)


    def forward(self, x):
        # Defines the forward pass

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.drop2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.drop3(x)

        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool4(x)
        x = self.drop4(x)

        x = self.flatten1(x)
        x = self.fc1(x)
        x = self.relu5(x)

        x = self.fc2(x)

        return x


class LSTMClassifier(nn.Module):
    def __init__(self, in_channels):
        super(LSTMClassifier, self).__init__()

        self.lstm = nn.LSTM(input_size=120, hidden_size=120)
        self.fc1 = nn.Linear(in_features=120, out_features=120)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(in_features=120, out_features=120)
        self.relu2 = nn.ReLU()
        
        self.fc3 = nn.Linear(in_features=120, out_features=1)

    def forward(self, x):

        out, (ht, ct) = self.lstm(x)
        out = out[:, -1, :]

        x = self.fc1(out)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)

        return x
    

if __name__ == "__main__":
    model_temp = NoiseDetector(p_dropout=0.4, fc_size=2048)
    X = torch.Tensor (np.ones((1024,1,120)))
    model_temp.forward(X)

    
    