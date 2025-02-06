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
from train_compensator import CompensationTrainer
from load_data_from_dicts import LoadDataFromDicts

if __name__ == "__main__":

    # define training hyperparameters
    INIT_LR = 5e-4
    BATCH_SIZE = 1024
    EPOCHS = 50
    CHOSEN_DATASET = "um"
    RANDOM_SEED = 31
    TEST_SIZE = 0.2
    COMPENSATOR_ARCH = "fcn-dae"

    # set the device we will be using to train the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    """
    # load the data from saved tensors
        tensorLoader = LoadDataFromDicts(chosen_dataset=CHOSEN_DATASET, 
                                        random_seed=RANDOM_SEED,
                                        test_size=TEST_SIZE)
        tensorLoader.load()
    
        classifier = ClassificationTrainer(lr=INIT_LR, batch_size=BATCH_SIZE, no_epochs=EPOCHS,
                                        X_train=tensorLoader.X_train,
                                        y_train=tensorLoader.y_train,
                                        X_test=tensorLoader.X_test,
                                        y_test=tensorLoader.y_test
                                        )"""

    X_train = torch.load("tensors/patient_based/um_train_X.pt")
    X_test = torch.load("tensors/patient_based/um_test_X.pt")
    X_train_reference = torch.load("tensors/patient_based/um_reference_train_X.pt")
    X_test_reference = torch.load("tensors/patient_based/um_reference_test_X.pt")

    y_train = torch.load("tensors/patient_based/um_train_y.pt")
    y_test = torch.load("tensors/patient_based/um_test_y.pt")

    
    classifier = ClassificationTrainer(lr=INIT_LR, batch_size=BATCH_SIZE, no_epochs=EPOCHS,
                                    X_train=X_train,
                                    y_train=y_train,
                                    X_test=X_test,
                                    y_test=y_test,
                                    X_test_reference=X_test_reference
                                    )
    
    classifier.train()
    results_train_acc, results_train_loss, results_val_acc, results_val_loss = classifier.getRawResults()
    best_confusion_matrix = classifier.getBestConfusionMatrix()

    """plotter = Plotter(dataset=CHOSEN_DATASET, seed=RANDOM_SEED, lr=INIT_LR, batch_size=BATCH_SIZE)
    plotter.plot_accuracy(results_train_acc,results_val_acc)
    plotter.plot_loss(results_train_loss, results_val_loss)
    plotter.plot_confusion_matrix(best_confusion_matrix)"""

    compensation_X_test, compensation_X_test_references = classifier.getCompensationSegments()


    compensator = CompensationTrainer(lr=INIT_LR,
                                      batch_size=BATCH_SIZE,
                                      no_epochs=EPOCHS,
                                      model_arch=COMPENSATOR_ARCH,
                                      X_train=X_train,
                                      y_train=X_train_reference, 
                                      X_test=compensation_X_test,
                                      y_test=compensation_X_test_references)
    compensator.train()
    
    comp_results_train_loss, comp_results_test_loss = compensator.getRawResults()

    plt.plot(comp_results_train_loss, label='Train Loss')
    plt.plot(comp_results_test_loss, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel("Loss")
    plt.title('Loss')
    plt.legend()
    plt.savefig(f"only-negatives-Normalized_Compensation_test_LOSS.png")
    plt.clf()
    
    for i in (range(200,1200,20)):
        try:
            compensator.getRandomSnapshot(random_seed=i)
        except:
            print(f"Snapshot failed for seed {i} ")

