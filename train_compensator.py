"""This file implements the compensator class."""

"""This file implements the training for given hyperparameters and train/test sets.
For the motion artifact compensation, it is a temporary idea to get the classification from
last epoch. Ideally there needs to be early stopping to detect overfit and take there."""

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
from compensation_models import DRDNN


class CompensationTrainer():
    def __init__(self, lr, batch_size, no_epochs, 
                 X_train, y_train, X_test, y_test,
                 model_name = "drdnn"):
        
        self.model_name = model_name
        self.lr = lr
        self.batch_size = batch_size
        self.no_epochs = no_epochs
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.trainData = CustomDataset(self.X_train, self.y_train)
        self.testData = CustomDataset(self.X_test, self.y_test)

        self.train_size = self.y_train.shape[0]
        self.test_size = self.y_test.shape[0]

        # initialize the train, validation, and test data loaders
        self.trainDataLoader = DataLoader(self.trainData, shuffle=True,batch_size=self.batch_size)
        self.testDataLoader = DataLoader(self.testData, shuffle=True, batch_size=self.batch_size)
       
        # number of steps per epoch 
        self.no_trainSteps = len(self.trainDataLoader.dataset) // self.batch_size
        self.no_testSteps = len(self.testDataLoader.dataset) // self.batch_size

        print(f"[INFO] Initializing model name:{self.model_name}...")
        
        if self.model_name == "drdnn":
            self.model = DRDNN(batch_size=self.batch_size).to(device=self.device)
       # more models can be added here for extensions 

        self.opt = optim.Adam(self.model.parameters(), lr=self.lr)
        self.lossFn = nn.MSELoss()

        # store the results
        self.results_train_acc = []
        self.results_train_loss = []
        self.results_val_acc = []
        self.results_val_loss = []
        self.conf_matrices_every_epoch =[]  
    
    def train(self):

        print("[INFO] Starting training...")
        startTime = time.time()

        # loop over epochs
        for e in tqdm(range(0, self.no_epochs)):

            # TRAIN 
            self.model.train()

            # initialize the total training and validation loss
            totalTrainLoss = 0
            totalValLoss = 0
            train_acc = 0
            val_acc = 0

            # loop over the training set in batches
            for X_batch, y_batch in self.trainDataLoader:
                X_batch,y_batch = torch.unsqueeze(X_batch,1),torch.unsqueeze(y_batch,1)

                # send the input to the device
                (X_batch, y_batch) = (X_batch.to(self.device), y_batch.to(self.device))

                self.opt.zero_grad()
                # perform a forward pass and calculate the training loss
                pred = self.model(X_batch)
                loss = self.lossFn(pred, y_batch)

                # add the loss to the total training loss so far and
                # calculate the number of correct predictions
                totalTrainLoss = totalTrainLoss + loss

                # zero out the gradients, perform the backpropagation step,
                # and update the weights
                loss.backward()
                self.opt.step()

            avgTrainLoss = float(totalTrainLoss /self.no_trainSteps)
            if e % 10 == 0:
                print(str.format("Epoch: {}, Avg training loss: {:.6f}", e+1, avgTrainLoss))

            # update our training history
            self.results_train_loss.append(avgTrainLoss)

            # EVAL
            # switch off autograd for evaluation
            with torch.no_grad():
                # set the model in evaluation mode
                self.model.eval()

    
                # loop over the validation set
                for X_batch, y_batch in self.testDataLoader:
                    X_batch, y_batch  = torch.unsqueeze(X_batch,1), torch.unsqueeze(y_batch,1)

                    # send the input to the device
                    (X_batch, y_batch) = (X_batch.to(self.device), y_batch.to(self.device))

                    # make the predictions and calculate the validation loss
                    pred = self.model(X_batch)
                    lossVal = self.lossFn(pred, y_batch)

                    totalValLoss = totalValLoss + lossVal


            avgValLoss = float(totalValLoss /self.no_testSteps)
            if e % 10 == 0:
                print(str.format("Epoch: {}, Avg Validation loss: {:.6f}", e+1, avgValLoss))

            # update our training history
            self.results_val_loss.append(avgValLoss)
    
        # finish measuring how long training took
        endTime = time.time()
        print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))

    def getRawResults(self):
        return (self.results_train_loss, self.results_val_loss)
