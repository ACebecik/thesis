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
from OLD_load_data_from_tensors import LoadDataFromTensor
from compensation_models import DRDNN, FCN_DAE, FCN_DAE_skip
import wandb


class CompensationTrainer():
    def __init__(self, lr, batch_size, no_epochs, 
                 X_train, y_train, X_test, y_test,
                 model_arch = "drdnn"):
        

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.no_epochs = no_epochs

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, run_config=None):
        
        with wandb.init(config=run_config):

            config = wandb.config

            self.model_name = config.COMPENSATOR_ARCH
            self.lr = config.INIT_LR
            self.batch_size = config.BATCH_SIZE
            self.lstm_hidden_size = config.LSTM_HIDDEN_SIZE

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
                self.model = DRDNN(lstm_hidden_size=self.lstm_hidden_size).to(device=self.device)
            
            elif self.model_name == "fcn-dae":
                self.model = FCN_DAE().to(device=self.device)

            elif self.model_name == "fcn-dae-skip":
                self.model = FCN_DAE_skip().to(device=self.device)

        # more models can be added here for extensions 

            self.opt = optim.Adam(self.model.parameters(), lr=self.lr)
            self.lossFn = nn.MSELoss()

            # store the results
            self.results_train_acc = []
            self.results_train_loss = []
            self.results_val_acc = []
            self.results_val_loss = []
            self.conf_matrices_every_epoch =[]  

            print("[INFO] Starting training...")
            startTime = time.time()

            scaler = MinMaxScaler()

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
                    if self.model_name == "drdnn":
                        pred = torch.unsqueeze(pred,dim=1)
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
                
                wandb.log({"epoch": e, "compensation_train_loss": avgTrainLoss})


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
                        if self.model_name == "drdnn":
                            pred = torch.unsqueeze(pred,dim=1)
                        
                        lossVal = self.lossFn(pred, y_batch)

                        totalValLoss = totalValLoss + lossVal


                avgValLoss = float(totalValLoss /self.no_testSteps)
                if e % 10 == 0:
                    print(str.format("Epoch: {}, Avg Validation loss: {:.6f}", e+1, avgValLoss))
                wandb.log({"epoch": e, "compensation_val_loss": avgValLoss})


                # update our training history
                self.results_val_loss.append(avgValLoss)
          
            # finish measuring how long training took
            endTime = time.time()
            print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))

    def getRawResults(self):
        return (self.results_train_loss, self.results_val_loss)


    def getRandomSnapshot(self,random_seed):
        
        X_snap = torch.unsqueeze(self.X_test[random_seed, :] , dim=0).to(device=self.device) 
        y_snap = torch.unsqueeze(self.y_test[random_seed, :] , dim=0).to(device=self.device)
            
        X_snap = torch.unsqueeze(X_snap , dim=0) 
        y_snap = torch.unsqueeze(y_snap , dim=0)

        y_pred_snap = self.model(X_snap)
        y_pred_snap = torch.squeeze(y_pred_snap)
        y_pred_snap = torch.unsqueeze(y_pred_snap, dim=1)
        
        scaler = MinMaxScaler()
        y_pred_snap = scaler.fit_transform(y_pred_snap.cpu().detach().numpy())

        X_snap = torch.squeeze(X_snap).cpu().numpy()
        y_snap = torch.squeeze(y_snap).cpu().numpy()
        

        plt.plot(y_pred_snap, label="compensated signal")
        plt.plot(X_snap, label="noisy signal" )
        plt.plot(y_snap, label="reference clean signal" )
        plt.legend()
        plt.savefig(f"snaps/snapshot of one segment seed{random_seed}snap.png")
        plt.clf()


if __name__ == "__main__":
    pass