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

            with open('dictionaries/new-1002/unovis_cecg_by_patients.pkl', 'rb') as f:
                unovis_noisy_ecg = pickle.load(f)

            with open('dictionaries/new-1002/unovis_cecg_labels_by_patients.pkl', 'rb') as f:
                unovis_noisy_ecg_labels = pickle.load(f)

            with open('dictionaries/new-1002/unovis_reference_ecg_by_patients.pkl', 'rb') as f:
                unovis_reference_ecg = pickle.load(f)

            joined_noisy_ecg = {**mit_noisy_ecg, **unovis_noisy_ecg}
            joined_noisy_ecg_labels = {**mit_noisy_ecg_labels, **unovis_noisy_ecg_labels}
            joined_reference_ecg = {**mit_reference_ecg, **unovis_reference_ecg}   

            random.seed(self.random_seed)
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
        torch.save(self.X_test, "tensors/new-1002/um_test_X.pt")
        torch.save(self.y_test, "tensors/new-1002/um_test_y.pt")
        torch.save(self.X_test_reference, "tensors/new-1002/um_reference_test_X.pt")
        torch.save(self.X_train, "tensors/new-1002/um_train_X.pt")  
        torch.save(self.y_train, "tensors/new-1002/um_train_y.pt")  
        torch.save(self.X_train_reference, "tensors/new-1002/um_reference_train_X.pt")

    def LoadSaveWithDatasetInfo(self):

       # gonna do another random here, will not be the same test keys with the other method
       # need to implement some kind of seeding for future
       #  ONLY WORKS FOR COMPENSATION FOR NOW

        if self.chosen_dataset == "um":
            with open('dictionaries/mit_noisy_ecg_by_patients.pkl', 'rb') as f:
                mit_noisy_ecg = pickle.load(f)

            with open('dictionaries/mit_noisy_ecg_labels_by_patients.pkl', 'rb') as f:
                mit_noisy_ecg_labels = pickle.load(f)
            
            with open('dictionaries/mit_reference_ecg_by_patients.pkl', 'rb') as f:
                mit_reference_ecg = pickle.load(f)

            with open('dictionaries/new-1002/unovis_cecg_by_patients.pkl', 'rb') as f:
                unovis_noisy_ecg = pickle.load(f)

            with open('dictionaries/new-1002/unovis_cecg_labels_by_patients.pkl', 'rb') as f:
                unovis_noisy_ecg_labels = pickle.load(f)

            with open('dictionaries/new-1002/unovis_reference_ecg_by_patients.pkl', 'rb') as f:
                unovis_reference_ecg = pickle.load(f)

            joined_noisy_ecg = {**mit_noisy_ecg, **unovis_noisy_ecg}
            joined_noisy_ecg_labels = {**mit_noisy_ecg_labels, **unovis_noisy_ecg_labels}
            joined_reference_ecg = {**mit_reference_ecg, **unovis_reference_ecg}

            random.seed(self.random_seed)
            test_keys = random.sample(joined_noisy_ecg.keys(), round(len(joined_noisy_ecg.keys())*self.test_size))

           # split compensation 
            self.X_test_unovis = np.zeros(120)
            self.X_test_mit = np.zeros(120)
            self.X_test_reference_unovis = np.zeros(120)
            self.X_test_reference_mit = np.zeros(120)

           # split classification 
            self.y_test_unovis = np.zeros(1)
            self.y_test_mit = np.zeros(1)

           # unified classification and compensation 
            self.X_train = np.zeros(120)
            self.y_train = np.zeros(1)
            self.X_train_reference = np.zeros(120)

            self.X_test = np.zeros(120)
            self.y_test = np.zeros(1)
            self.X_test_reference = np.zeros(120)

            for key in joined_noisy_ecg.keys():

               # if test set and from mit dataset 
                if key in test_keys and key > 1000:
                    self.X_test_mit = np.vstack((self.X_test_mit, joined_noisy_ecg[key]))
                    self.y_test_mit = np.concatenate((self.y_test_mit,joined_noisy_ecg_labels[key]))
                    self.X_test_reference_mit = np.vstack((self.X_test_reference_mit, joined_reference_ecg[key]))
                    
                    self.X_test = np.vstack((self.X_test, joined_noisy_ecg[key]))
                    self.y_test = np.concatenate((self.y_test, joined_noisy_ecg_labels[key]))
                    self.X_test_reference = np.vstack((self.X_test_reference, joined_reference_ecg[key]))
            
               # if test set and from unovis dataset 
                elif key in test_keys and key < 1000:
                    self.X_test_unovis = np.vstack((self.X_test_unovis, joined_noisy_ecg[key]))
                    self.y_test_unovis = np.concatenate((self.y_test_unovis,joined_noisy_ecg_labels[key]))
                    self.X_test_reference_unovis = np.vstack((self.X_test_reference_unovis, joined_reference_ecg[key]))

                    self.X_test = np.vstack((self.X_test, joined_noisy_ecg[key]))
                    self.y_test = np.concatenate((self.y_test, joined_noisy_ecg_labels[key]))
                    self.X_test_reference = np.vstack((self.X_test_reference, joined_reference_ecg[key]))
            
              # if train set  
                else:
                    self.X_train = np.vstack((self.X_train, joined_noisy_ecg[key]))
                    self.y_train = np.concatenate((self.y_train, joined_noisy_ecg_labels[key]))
                    self.X_train_reference = np.vstack((self.X_train_reference, joined_reference_ecg[key]))

            self.X_test_unovis = torch.Tensor(np.delete(self.X_test_unovis, (0),axis=0 )) 
            self.X_test_mit = torch.Tensor(np.delete(self.X_test_mit, (0),axis=0 )) 
            self.X_test_reference_unovis = torch.Tensor(np.delete(self.X_test_reference_unovis, (0),axis=0 )) 
            self.X_test_reference_mit = torch.Tensor(np.delete(self.X_test_reference_mit, (0),axis=0 )) 

            self.y_test_unovis = torch.Tensor(np.delete(self.y_test_unovis, (0),axis=0 )) 
            self.y_test_mit = torch.Tensor(np.delete(self.y_test_mit, (0),axis=0 )) 

            self.X_train = torch.Tensor(np.delete(self.X_train, (0),axis=0 ) )
            self.y_train = torch.Tensor(np.delete(self.y_train, (0),axis=0 ) )
            self.X_train_reference = torch.Tensor(np.delete(self.X_train_reference, (0),axis=0 ) )    
            
            self.X_test = torch.Tensor(np.delete(self.X_test, (0),axis=0 ) )
            self.y_test = torch.Tensor(np.delete(self.y_test, (0),axis=0 ) )
            self.X_test_reference = torch.Tensor(np.delete(self.X_test_reference, (0),axis=0 ) )    


            torch.save(self.X_test_unovis, "tensors/um/um_test_X_unovis.pt")
            torch.save(self.X_test_mit, "tensors/um/um_test_X_mit.pt")
            torch.save(self.X_test_reference_unovis, "tensors/um/um_reference_test_X_unovis.pt")
            torch.save(self.X_test_reference_mit, "tensors/um/um_reference_test_X_mit.pt")

            torch.save(self.y_test_unovis, "tensors/um/um_test_y_unovis.pt")
            torch.save(self.y_test_mit, "tensors/um/um_test_y_mit.pt")

            torch.save(self.X_train, "tensors/um/um_train_X.pt")  
            torch.save(self.y_train, "tensors/um/um_train_y.pt")  
            torch.save(self.X_train_reference, "tensors/um/um_reference_train_X.pt")
    

            torch.save(self.X_test, "tensors/um/um_test_X.pt")  
            torch.save(self.y_test, "tensors/um/um_test_y.pt")  
            torch.save(self.X_test_reference, "tensors/um/um_reference_test_X.pt")
        else:
            pass

if __name__ =="__main__":
    loader = LoadDataFromDicts()
    loader.LoadSaveWithDatasetInfo()
    try:
        print(loader.X_test_unovis.shape, loader.y_test_unovis.shape, loader.X_test_reference_unovis.shape)
        print(loader.X_test_mit.shape, loader.y_test_mit.shape, loader.X_test_reference_mit.shape)
        print(loader.X_test.shape, loader.y_test.shape, loader.X_test_reference.shape)
        print(loader.X_train.shape, loader.y_train.shape, loader.X_train_reference.shape)
    except:
        print("something went wrong")


