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
from train import ClassificationTrainer
from plotter import Plotter
from train_compensator import CompensationTrainer
from load_data_from_dicts import LoadDataFromDicts
import wandb
import pprint
from compensation_models import DRDNN, FCN_DAE, FCN_DAE_skip
from classification_models import NoiseDetector, LSTMClassifier


def loadModel(model_name, device):
    if model_name == "drdnn":
        model = DRDNN(lstm_hidden_size=360).to(device=device)
    
    elif model_name == "fcn-dae":
        model = FCN_DAE().to(device=device)

    elif model_name == "fcn-dae-skip":
        model = FCN_DAE_skip().to(device=device)

    elif model_name == "ansari":
        model = NoiseDetector(p_dropout=0.2, fc_size=2048).to(device=device)

    elif model_name == "lstm":
        model = LSTMClassifier(config_hidden_size=120).to(device=device)
    
    model.load_state_dict(torch.load(f"models/{model_name}.pt"))
    return model

def testClassifier(model_name, model, X_test, y_test):
    
    if model_name == "ansari":
        batch_size = 1728
    else:
        batch_size = 4416

    testData = CustomDataset(X_test, y_test)
    test_size = y_test.shape[0]

    # initialize the test data loader
    testDataLoader = DataLoader(testData, shuffle=True, batch_size=batch_size)

    # number of steps per epoch 
    no_testSteps = len(testDataLoader.dataset) // batch_size
    with torch.no_grad():
        # set the model in evaluation mode
        model.eval()
        totalTestLoss = 0
        test_acc = 0

        # clear confusion matrix
        test_conf_matrix =np.zeros((2,2)) 
        batch_counter = 0
        lossFn = nn.BCEWithLogitsLoss()

        # loop over the validation set
        for X_batch, y_batch in testDataLoader:
            X_batch, y_batch  = torch.unsqueeze(X_batch,1), torch.unsqueeze(y_batch,1)

            # send the input to the device
            (X_batch, y_batch) = (X_batch.to(device), y_batch.to(device))

            # make the predictions and calculate the validation loss
            pred = model(X_batch)
            lossTest = lossFn(pred, y_batch)
            getPreds = nn.Sigmoid()
            predProbTest = getPreds(pred)
            totalTestLoss = totalTestLoss + lossTest
            test_acc = test_acc + ((predProbTest>0.5)==y_batch).sum()

        # confusion matrix 
            predictions = (predProbTest>0.5)*1
            temp_conf_matrix = confusion_matrix(y_batch.cpu().numpy(), predictions.cpu().numpy())
            test_conf_matrix = np.add(test_conf_matrix, temp_conf_matrix)



    avgTestAcc = float(test_acc/test_size)
    avgTestLoss = float(totalTestLoss /no_testSteps)
    print(str.format("Avg Test Loss: {:.6f}, Avg Test Acc: {:.6f}", avgTestLoss, avgTestAcc))

    return avgTestLoss, avgTestAcc, test_conf_matrix 

if __name__ == "__main__":

    X_test = torch.load("tensors/final_tensors_1703/um_test_X.pt")
    y_test = torch.load("tensors/final_tensors_1703/um_test_y.pt")
    X_test_reference = torch.load("tensors/final_tensors_1703/um_reference_test_X.pt")


    X_test_reference_mit = torch.load("tensors/final_tensors_1703/mit_reference_test_X.pt")
    X_test_mit = torch.load("tensors/final_tensors_1703/mit_test_X.pt")
    y_test_mit = torch.load("tensors/final_tensors_1703/mit_test_y.pt")

    X_test_unovis = torch.load("tensors/final_tensors_1703/unovis_test_X.pt")
    X_test_reference_unovis = torch.load("tensors/final_tensors_1703/unovis_reference_test_X.pt")
    y_test_unovis = torch.load("tensors/final_tensors_1703/unovis_test_y.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    ansari_model = loadModel("ansari", device=device)
    fcn_dae_model = loadModel("fcn-dae", device=device)

    testClassifier("ansari", ansari_model, X_test=X_test_unovis, y_test=y_test_unovis)




