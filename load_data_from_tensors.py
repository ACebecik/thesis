"""This file implements the loads from saved tensors for the experiment case."""
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

class LoadDataFromTensor():
    def __init__(self, chosen_dataset = "augmented_um", random_seed = 31, test_size = 0.2):
        self.chosen_dataset = chosen_dataset
        self.random_seed = random_seed
        self.test_size = test_size

    
    def load(self):
        if self.chosen_dataset == "augmented_um":
            self.X_train = torch.load("tensors/augmented_UM_train_X.pt")
            self.y_train = torch.load("tensors/augmented_UM_train_y.pt")
            self.X_test = torch.load ("tensors/augmented_UM_test_X.pt")
            self.y_test = torch.load("tensors/augmented_UM_test_y.pt")
        
        elif self.chosen_dataset == 'um':
            X_mit = torch.load("tensors/mit_all_records_X_w120_fixed.pt")
            y_mit = torch.load("tensors/mit_all_records_y_w360.pt")

            X_unovis = torch.load("tensors/unovis_all_records_X_w120.pt")
            y_unovis = torch.load("tensors/unovis_all_records_y_w120.pt")

            X = np.vstack((X_mit, X_unovis))
            y = np.concatenate((y_mit, y_unovis))

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_seed)      
        
        elif self.chosen_dataset == 'mit':
            X_mit = torch.load("tensors/mit_all_records_X_w120_fixed.pt")
            y_mit = torch.load("tensors/mit_all_records_y_w360.pt")

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_seed)    

        elif self.chosen_dataset == "unovis":
            X_unovis = torch.load("tensors/unovis_all_records_X_w120.pt")
            y_unovis = torch.load("tensors/unovis_all_records_y_w120.pt")

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_seed)  
            
