"""This file implements the compensation models for motion artifacts. 
Implemented models:
1- DRDNN method from Antczak, 2018
2- Fully Conv Net DAE from Chiang, 2019"""


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
        self.fc1 = nn.Linear(in_features=120, out_features=120)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(in_features=120, out_features=120)
        self.relu2 = nn.ReLU()
        
        self.fc3 = nn.Linear(in_features=120, out_features=120)

    def forward(self, x):

        out, (ht, ct) = self.lstm(x)
        out = out[:, -1, :]

        x = self.fc1(out)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)

        return x
    


class FCN_DAE(nn.Module):
    def __init__(self, in_channels=1):
        super(FCN_DAE, self).__init__()

        # First set of Conv,Relu,Pooling,Dropout
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=20, kernel_size=16, stride=2, padding=7)
        self.relu1 = nn.ReLU()

        # 2nd
        self.conv2 = nn.Conv1d(in_channels=20, out_channels=40, kernel_size=16, stride=2, padding=7)
        self.relu2 = nn.ReLU()

        #3rd
        self.conv3 = nn.Conv1d(in_channels=40, out_channels=80, kernel_size=16, stride=2, padding=7)
        self.relu3 = nn.ReLU()

        #4th
        self.conv4 = nn.Conv1d(in_channels=80, out_channels=160, kernel_size=8, stride=1, padding=1)
        self.relu4 = nn.ReLU()

        # decoder

        self.deconv1 = nn.ConvTranspose1d(in_channels=160, out_channels=80, kernel_size=8, stride=1, padding=1)
        self.relu5 = nn.ReLU()

        self.deconv2 = nn.ConvTranspose1d(in_channels=80, out_channels=40, kernel_size=16, stride=2, padding=7)
        self.relu6 = nn.ReLU()

        self.deconv3 = nn.ConvTranspose1d(in_channels=40, out_channels=20, kernel_size=16, stride=2, padding=7)
        self.relu7 = nn.ReLU()

        self.deconv4 = nn.ConvTranspose1d(in_channels=20, out_channels=1, kernel_size=16, stride=2, padding=7)

    def forward(self, x):
        # Defines the forward pass

        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.relu4(x)

        x = self.deconv1(x)
        x = self.relu5(x)

        x = self.deconv2(x)
        x = self.relu6(x)

        x = self.deconv3(x)
        x = self.relu7(x)

        x = self.deconv4(x)

        return x


if __name__ == "__main__":
    model_temp = FCN_DAE()
    X = torch.Tensor (np.ones((1024,1,120)))
    model_temp.forward(X)

    
    