"""This file implements the CustomDataset class used to feed network the data via DataLoader."""
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

class CustomDataset(Dataset):
    def __init__(self, x, y):
        # data loading
        self.x = x
        self.y = y

    def __getitem__(self, index):
        # allows for indexing
        get_x = self.x[index]
        get_y = self.y[index]
        return get_x,get_y

    def __len__(self):
        return len(self.x)