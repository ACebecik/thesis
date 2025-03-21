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
from classification_models import NoiseDetector, LSTMClassifier
from OLD_load_data_from_tensors import LoadDataFromTensor
import wandb



class ClassificationTrainer():
    def __init__(self, lr, batch_size, no_epochs, 
                 X_train, y_train, X_test, y_test, X_test_reference,
                 model_name = "ansari"):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.no_epochs = no_epochs
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X_test_reference = X_test_reference

    
    def train(self, run_config=None):
    
        with wandb.init(config=run_config):

            config = wandb.config
            
            self.model_name = config.CLASSIFIER_ARCH
            self.lr = config.INIT_LR
            self.batch_size = config.BATCH_SIZE
            self.model_dropout = config.DROPOUT

            self.compensation_X_test_segments = [] 
            
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
            
            if self.model_name == "ansari":
                self.model = NoiseDetector(in_channels=1, p_dropout=self.model_dropout).to(device=self.device)
        # more models can be added here for extensions 

            elif self.model_name == "lstm":
                self.model = LSTMClassifier(in_channels=1).to(device=self.device)

            self.opt = optim.Adam(self.model.parameters(), lr=self.lr)
            self.lossFn = nn.BCEWithLogitsLoss()

            # store the results
            self.results_train_acc = []
            self.results_train_loss = []
            self.results_val_acc = []
            self.results_val_loss = []
            self.conf_matrices_every_epoch =[] 

            self.compensation_segment_idxs = torch.empty(0).to(device=self.device)

            compensation_X_test_segments = [] 
            corresponding_X_test_reference = []   

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
                    getPreds = nn.Sigmoid()
                    predProb = getPreds(pred)
                    train_acc = train_acc +  ((predProb>0.5) ==y_batch).sum() # correctly predicted samples

                    # add the loss to the total training loss so far and
                    # calculate the number of correct predictions
                    totalTrainLoss = totalTrainLoss + loss

                    # zero out the gradients, perform the backpropagation step,
                    # and update the weights
                    loss.backward()
                    self.opt.step()

                avgTrainAcc = float(train_acc/self.train_size)
                avgTrainLoss = float(totalTrainLoss /self.no_trainSteps)
                if e % 10 == 0:
                    print(str.format("Epoch: {}, Avg training loss: {:.6f}, Avg Train Acc: {:.6f}", e+1, avgTrainLoss, avgTrainAcc))

                wandb.log({"epoch": e, "classification_train_loss": avgTrainLoss, "classification_train_acc": avgTrainAcc})


                # update our training history
                self.results_train_acc.append(avgTrainAcc)
                self.results_train_loss.append(avgTrainLoss)

                # EVAL
                # switch off autograd for evaluation
                with torch.no_grad():
                    # set the model in evaluation mode
                    self.model.eval()

                    # clear confusion matrix
                    conf_matrix =np.zeros((2,2)) 
                    batch_counter = 0
        
                    # loop over the validation set
                    for X_batch, y_batch in self.testDataLoader:
                        X_batch, y_batch  = torch.unsqueeze(X_batch,1), torch.unsqueeze(y_batch,1)

                        # send the input to the device
                        (X_batch, y_batch) = (X_batch.to(self.device), y_batch.to(self.device))

                        # make the predictions and calculate the validation loss
                        pred = self.model(X_batch)
                        lossVal = self.lossFn(pred, y_batch)
                        predProbVal = getPreds(pred)
                        totalValLoss = totalValLoss + lossVal
                        val_acc = val_acc + ((predProbVal>0.5)==y_batch).sum()

                    # confusion matrix 
                        predictions = (predProbVal>0.5)*1
                        temp_conf_matrix = confusion_matrix(y_batch.cpu().numpy(), predictions.cpu().numpy())
                        conf_matrix = np.add(conf_matrix, temp_conf_matrix)
                    
                        # based on predictions choose where to compensate artifacts
                        if e+1 == self.no_epochs:

                            self.zero_indices = np.where(predictions.cpu().numpy() == 0)[0] 


                            for idx in self.zero_indices:
                                compensation_X_test_segments.append(np.array(X_batch[idx,:,:].cpu()))
                                real_idx = batch_counter * self.batch_size + idx
                                corresponding_X_test_reference.append(np.array(self.X_test_reference[real_idx,:].cpu()))
                            
                            batch_counter = batch_counter + 1
                            



                avgValAcc = float(val_acc/self.test_size)
                avgValLoss = float(totalValLoss /self.no_testSteps)
                if e % 10 == 0:
                    print(str.format("Epoch: {}, Avg Validation loss: {:.6f}, Avg Val Acc: {:.6f}", e+1, avgValLoss, avgValAcc))

                wandb.log({"epoch": e, "classification_val_loss": avgValLoss, "classification_val_acc": avgValAcc})


                # update our training history
                self.results_val_acc.append(avgValAcc)
                self.results_val_loss.append(avgValLoss)
                self.conf_matrices_every_epoch.append(conf_matrix)
        
            # finish measuring how long training took
            endTime = time.time()
            print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))

            # convert the compensation segments
            compensation_X_test_segments_np = np.array(compensation_X_test_segments)
            corresponding_X_test_reference_np = np.array(corresponding_X_test_reference)

            self.compensation_X_test_segments = torch.Tensor(compensation_X_test_segments_np)
            self.compensation_X_test_reference = torch.Tensor(corresponding_X_test_reference_np)

            self.compensation_X_test_segments = torch.squeeze(self.compensation_X_test_segments)

    def getRawResults(self):
        return (self.results_train_acc, self.results_train_loss, self.results_val_acc, self.results_val_loss)

    def getBestConfusionMatrix(self):
        self.best_epoch = np.argmax(self.results_val_acc)
        return self.conf_matrices_every_epoch[self.best_epoch]
    
    def getBestConfusionMatrixParameters(self):
        tn = self.conf_matrices_every_epoch[self.best_epoch][0,0]   
        tp = self.conf_matrices_every_epoch[self.best_epoch][1,1]
        fp = self.conf_matrices_every_epoch[self.best_epoch][0,1]
        fn = self.conf_matrices_every_epoch[self.best_epoch][1,0]

        self.accuracy = (tp+tn) / (tp+fp+tn+fn)
        self.specificity = tn / (tn+fp)
        self.sensitivity = tp / (tp+fn) 

        return self.accuracy, self.specificity, self.sensitivity
    
    def getCompensationSegments(self):

        return self.compensation_X_test_segments, self.compensation_X_test_reference

