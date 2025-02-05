"""This file implements the loads from saved dicts for the experiment case."""

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
import random

class LoadDataFromDicts():
    def __init__(self, chosen_dataset = "um", random_seed = 31, test_size = 0.2):
        self.chosen_dataset = chosen_dataset
        self.random_seed = random_seed
        self.test_size = test_size

    
    def load(self):
        if self.chosen_dataset == "augmented_um":
            pass
        
        elif self.chosen_dataset == 'um':
            with open('dictionaries/mit_noisy_ecg_by_patients.pkl', 'rb') as f:
                mit_noisy_ecg = pickle.load(f)

            with open('dictionaries/mit_noisy_ecg_labels_by_patients.pkl', 'rb') as f:
                mit_noisy_ecg_labels = pickle.load(f)
            
            with open('dictionaries/mit_reference_ecg_by_patients.pkl', 'rb') as f:
                mit_reference_ecg = pickle.load(f)

            with open('dictionaries/unovis_cecg_by_patients.pkl', 'rb') as f:
                unovis_noisy_ecg = pickle.load(f)

            with open('dictionaries/unovis_cecg_labels_by_patients.pkl', 'rb') as f:
                unovis_noisy_ecg_labels = pickle.load(f)

            with open('dictionaries/unovis_reference_ecg_by_patients.pkl', 'rb') as f:
                unovis_reference_ecg = pickle.load(f)

            joined_noisy_ecg = {**mit_noisy_ecg, **unovis_noisy_ecg}
            joined_noisy_ecg_labels = {**mit_noisy_ecg_labels, **unovis_noisy_ecg_labels}
            joined_reference_ecg = {**mit_reference_ecg, **unovis_reference_ecg}   

            test_keys = random.sample(joined_noisy_ecg.keys(), round(len(joined_noisy_ecg.keys())*self.test_size))

            self.X_test = np.zeros(120)
            self.y_test = np.zeros(1)
            self.X_train = np.zeros(120)
            self.y_train = np.zeros(1)
            self.X_test_reference = np.zeros(120)
            self.X_train_reference = np.zeros(120)

            for key in joined_noisy_ecg.keys():
                if key in test_keys:
                    self.X_test = np.vstack((self.X_test, joined_noisy_ecg[key]))
                    self.y_test = np.concatenate((self.y_test,joined_noisy_ecg_labels[key]))
                    self.X_test_reference = np.vstack((self.X_test_reference, joined_reference_ecg[key]))
                else:
                    self.X_train = np.vstack((self.X_train, joined_noisy_ecg[key]))
                    self.y_train = np.concatenate((self.y_train, joined_noisy_ecg_labels[key]))
                    self.X_train_reference = np.vstack((self.X_train_reference, joined_reference_ecg[key]))

            self.X_test = torch.Tensor(np.delete(self.X_test, (0),axis=0 )) 
            self.y_test = torch.Tensor(np.delete(self.y_test, (0),axis=0 )) 
            self.X_test_reference = torch.Tensor(np.delete(self.X_test_reference, (0),axis=0 )) 
            self.X_train = torch.Tensor(np.delete(self.X_train, (0),axis=0 ) )
            self.y_train = torch.Tensor(np.delete(self.y_train, (0),axis=0 ) )
            self.X_train_reference = torch.Tensor(np.delete(self.X_train_reference, (0),axis=0 ) )        
        
        elif self.chosen_dataset == 'mit':
            pass  

        elif self.chosen_dataset == "unovis":
            pass 
            
    def loadClean(self):
        if self.chosen_dataset == "augmented_um":
            pass
        
        elif self.chosen_dataset == 'um':
            pass    
        
        elif self.chosen_dataset == 'mit':
            pass 

        elif self.chosen_dataset == "unovis":
            pass 
    
    def SaveToTensors(self):
        torch.save(self.X_test, "tensors/patient_based/um_test_X.pt")
        torch.save(self.y_test, "tensors/patient_based/um_test_y.pt")
        torch.save(self.X_test_reference, "tensors/patient_based/um_reference_test_X.pt")
        torch.save(self.X_train, "tensors/patient_based/um_train_X.pt")  
        torch.save(self.y_train, "tensors/patient_based/um_train_y.pt")  
        torch.save(self.X_train_reference, "tensors/patient_based/um_reference_train_X.pt") 

if __name__ =="__main__":
    loader = LoadDataFromDicts()
    loader.load()
    loader.SaveToTensors()
    print(loader.X_test.shape, loader.y_test.shape, loader.X_train.shape, loader.y_train.shape)


