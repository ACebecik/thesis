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

if __name__ == "__main__":

    # define training hyperparameters
    INIT_LR = 5e-4
    BATCH_SIZE = 8192
    EPOCHS = 300
    CHOSEN_DATASET = "augmented_um"
    RANDOM_SEED = 31
    TEST_SIZE = 0.2

    # set the device we will be using to train the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   # load the data from saved tensors
    """
    if CHOSEN_DATASET == "augmented_um":
        X_train = torch.load("tensors/augmented_UM_train_X.pt")
        y_train = torch.load("tensors/augmented_UM_train_y.pt")
        X_test = torch.load ("tensors/augmented_UM_test_X.pt")
        y_test = torch.load("tensors/augmented_UM_test_y.pt")
    
    elif CHOSEN_DATASET == 'um':
        X_mit = torch.load("tensors/mit_all_records_X_w120_fixed.pt")
        y_mit = torch.load("tensors/mit_all_records_y_w360.pt")

        X_unovis = torch.load("tensors/unovis_all_records_X_w120.pt")
        y_unovis = torch.load("tensors/unovis_all_records_y_w120.pt")

        X = np.vstack((X_mit, X_unovis))
        y = np.concatenate((y_mit, y_unovis))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=31)      
    
    elif CHOSEN_DATASET == 'mit':
        X_mit = torch.load("tensors/mit_all_records_X_w120_fixed.pt")
        y_mit = torch.load("tensors/mit_all_records_y_w360.pt")

        X_train, X_test, y_train, y_test = train_test_split(
            X_mit, y_mit, test_size=0.2, random_state=31)    

    elif CHOSEN_DATASET == "unovis":
        X_unovis = torch.load("tensors/unovis_all_records_X_w120.pt")
        y_unovis = torch.load("tensors/unovis_all_records_y_w120.pt")

        X_train, X_test, y_train, y_test = train_test_split(
            X_unovis, y_unovis, test_size=0.2, random_state=31)
    """     
    
    tensorLoader = LoadDataFromTensor(chosen_dataset=CHOSEN_DATASET, 
                                      random_seed=RANDOM_SEED,
                                      test_size=TEST_SIZE)
    tensorLoader.load()
    
    trainData = CustomDataset(tensorLoader.X_train, tensorLoader.y_train)
    testData = CustomDataset(tensorLoader.X_test, tensorLoader.y_test)
    train_size = tensorLoader.y_train.shape[0]
    test_size = tensorLoader.y_test.shape[0]  

    # initialize the train, validation, and test data loaders
    trainDataLoader = DataLoader(trainData, shuffle=True,batch_size=BATCH_SIZE)
    testDataLoader = DataLoader(testData, shuffle=True, batch_size=BATCH_SIZE)
    # calculate steps per epoch for training and validation set
    trainSteps = len(trainDataLoader.dataset) // BATCH_SIZE
    testSteps = len(testDataLoader.dataset) // BATCH_SIZE

    print("Initializing model...")
    model = NoiseDetector(in_channels=1).to(device)
    opt = optim.Adam(model.parameters(), lr=INIT_LR)
    lossFn = nn.BCEWithLogitsLoss()

    # initialize a dictionary to store training history
    H = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    results_train_acc = []
    results_train_loss = []
    results_val_acc = []
    results_val_loss = []

    conf_matrices_every_epoch =[] 


    # measure how long training is going to take
    print("[INFO] training the network...")
    startTime = time.time()


    # loop over epochs
    for e in tqdm(range(0, EPOCHS)):

        # TRAINING
        # train the model
        model.train()

        # initialize the total training and validation loss
        totalTrainLoss = 0
        totalValLoss = 0
        train_acc = 0
        val_acc = 0


        # loop over the training set in batches
        for X_batch, y_batch in trainDataLoader:
            X_batch,y_batch = torch.unsqueeze(X_batch,1),torch.unsqueeze(y_batch,1)
            #X_batch,y_batch = torch.unsqueeze(X_batch,0), torch.unsqueeze(y_batch,0)

            """            # break if end of dataset
                        if X_batch.shape[-1] < BATCH_SIZE:
                            break"""

            # send the input to the device
            (X_batch, y_batch) = (X_batch.to(device), y_batch.to(device))

            opt.zero_grad()
            # perform a forward pass and calculate the training loss
            pred = model(X_batch)
            #print(pred.shape, y.shape)
            loss = lossFn(pred, y_batch)
            getPreds = nn.Sigmoid()
            predProb = getPreds(pred)

            train_acc = train_acc +  ((predProb>0.5) ==y_batch).sum() # correctly predicted samples

            # add the loss to the total training loss so far and
            # calculate the number of correct predictions
            totalTrainLoss = totalTrainLoss + loss


            # zero out the gradients, perform the backpropagation step,
            # and update the weights
            loss.backward()
            opt.step()

        avgTrainAcc = float(train_acc/train_size)
        avgTrainLoss = float(totalTrainLoss /trainSteps)
        if e % 10 == 0:
            print(str.format("Epoch: {}, Avg training loss: {:.6f}, Avg Train Acc: {:.6f}", e+1, avgTrainLoss, avgTrainAcc))

        # update our training history
        results_train_acc.append(avgTrainAcc)
        results_train_loss.append(avgTrainLoss)


        #EVAL
        # switch off autograd for evaluation
        with torch.no_grad():
            # set the model in evaluation mode
            model.eval()

            # clear confusion matrix
            conf_matrix =np.zeros((2,2)) 
 
            # loop over the validation set
            for X_batch, y_batch in testDataLoader:
                X_batch, y_batch  = torch.unsqueeze(X_batch,1), torch.unsqueeze(y_batch,1)

                """                # break if end of dataset
                                if x.shape[-1] < BATCH_SIZE:
                                    break"""

                # send the input to the device
                (X_batch, y_batch) = (X_batch.to(device), y_batch.to(device))

                # make the predictions and calculate the validation loss
                pred = model(X_batch)
                lossVal = lossFn(pred, y_batch)

                predProbVal = getPreds(pred)

                totalValLoss = totalValLoss + lossVal

                val_acc = val_acc + ((predProbVal>0.5)==y_batch).sum()

               # confusion matrix 
                predictions = (predProbVal>0.5)*1
                temp_conf_matrix = confusion_matrix(y_batch.cpu().numpy(), predictions.cpu().numpy())
                conf_matrix = np.add(conf_matrix, temp_conf_matrix)

        avgValAcc = float(val_acc/test_size)
        avgValLoss = float(totalValLoss /testSteps)
        if e % 10 == 0:
            print(str.format("Epoch: {}, Avg Validation loss: {:.6f}, Avg Val Acc: {:.6f}", e+1, avgValLoss, avgValAcc))

        # update our training history
        results_val_acc.append(avgValAcc)
        results_val_loss.append(avgValLoss)
        conf_matrices_every_epoch.append(conf_matrix)

    # finish measuring how long training took
    endTime = time.time()
    print("[INFO] total time taken to train the model: {:.2f}s".format(
        endTime - startTime))


    plt.plot(results_train_acc, label='Train Acc')
    plt.plot(results_val_acc, label='Val Acc')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"plots/augmented_UM_all_Acc_plot_w120_lr{INIT_LR}_batchsize{BATCH_SIZE}.png")
    plt.clf()

    plt.plot(results_train_loss, label='Train Loss')
    plt.plot(results_val_loss, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel("Loss")
    plt.title('Loss')
    plt.legend()
    plt.savefig(f"plots/augmented_UM_all_Loss_plot_w120_lr{INIT_LR}_batchsize{BATCH_SIZE}.png")
    plt.clf()

   #display confusion matrix for the best accuracy epoch
    best_epoch = np.argmax(results_val_acc)
    disp_conf_matrix = ConfusionMatrixDisplay(conf_matrices_every_epoch[best_epoch])
    disp_conf_matrix.plot()
    plt.savefig(f"plots/augmented_UM_all_Conf_Matrix_w120_lr{INIT_LR}_batchsize{BATCH_SIZE}.png")
    plt.clf()

   # store log of experiment
