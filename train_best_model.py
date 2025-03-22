import wandb
import pandas as pd
import matplotlib.pyplot as plt
from classification_models import NoiseDetector, LSTMClassifier
from compensation_models import DRDNN, FCN_DAE, FCN_DAE_skip
import torch
import numpy as np
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


class newconfig:
    def __init__(self, classifier_arch, compensator_arch, lr, batch_size, dropout,
                 ansari_hidden_size, lstm_hidden_size):
        self.CLASSIFIER_ARCH =classifier_arch
        self.COMPENSATOR_ARCH = compensator_arch
        self.INIT_LR = lr
        self.BATCH_SIZE = batch_size
        self.DROPOUT = dropout
        self.ANSARI_HIDDEN_SIZE = ansari_hidden_size
        self.LSTM_HIDDEN_SIZE = lstm_hidden_size


api = wandb.Api()
wandb.init(project="lstm-best-run-newconfig-class")
sweep = api.sweep("alperencebecik-rwth-aachen-university/lstm-aum-hidden-size-optimization/mmwdysta")

# Get best run parameters
best_run = sweep.best_run(order="classification_val_acc")

best_parameters = best_run.config
model_name_selected = best_parameters["CLASSIFIER_ARCH"]

run_config = best_parameters
print(best_parameters)

run_config["EPOCHS"] = 10 
"""
sweep_config ={
    "method": "random",
    "program": "train_best_model.py" 
} 

metric ={
    # "name": "compensation_val_loss",
    "name": "classification_val_loss",
    "goal": "minimize"
} 


parameters_dict = {
    "CLASSIFIER_ARCH":{
        # "values" : ["lstm", "ansari"]  
        "values" : run_config["CLASSIFIER_ARCH"]  
    } ,
    'COMPENSATOR_ARCH': {
        #  'values': ['fcn-dae', "fcn-dae-skip", "drdnn"]
        "values" : run_config[ "COMPENSATOR_ARCH"]  
            },
    'INIT_LR': {
        "values" : run_config["INIT_LR"] 
        },
    "BATCH_SIZE":{
        "values" : run_config["BATCH_SIZE"] 
    },
    "DROPOUT":{
        "values" : run_config["DROPOUT"]  
    },
    "ANSARI_HIDDEN_SIZE":{
        "values": run_config["ANSARI_HIDDEN_SIZE"]  
    },
    "LSTM_HIDDEN_SIZE":{
        "values": run_config["LSTM_HIDDEN_SIZE"] 
    }  
} 

sweep_config["metric"] = metric

sweep_config['parameters'] = parameters_dict

pprint.pprint(sweep_config)"""

best_config = newconfig(classifier_arch= best_parameters["CLASSIFIER_ARCH"],
                        compensator_arch= best_parameters["COMPENSATOR_ARCH"],
                        lr=best_parameters["INIT_LR"],
                        batch_size= best_parameters["BATCH_SIZE"],
                        dropout=best_parameters["DROPOUT"],
                        ansari_hidden_size= best_parameters["ANSARI_HIDDEN_SIZE"],
                        lstm_hidden_size=best_parameters["LSTM_HIDDEN_SIZE"]     )


# set the device we will be using to train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train = torch.load("tensors/final_tensors_1703/augmented_um_train_X.pt")
X_test = torch.load("tensors/final_tensors_1703/um_test_X.pt")
X_train_reference = torch.load("tensors/final_tensors_1703/augmented_um_reference_train_X.pt")
X_test_reference = torch.load("tensors/final_tensors_1703/um_reference_test_X.pt")

y_train = torch.load("tensors/final_tensors_1703/augmented_um_train_y.pt")
y_test = torch.load("tensors/final_tensors_1703/um_test_y.pt")

X_validation = torch.load("tensors/final_tensors_1703/um_validation_X.pt")
y_validation = torch.load("tensors/final_tensors_1703/um_validation_y.pt")
X_validation_reference = torch.load("tensors/final_tensors_1703/um_reference_validation_X.pt")    

X_test_reference_mit = torch.load("tensors/final_tensors_1703/mit_reference_test_X.pt")
X_test_mit = torch.load("tensors/final_tensors_1703/mit_test_X.pt")

X_test_unovis = torch.load("tensors/final_tensors_1703/unovis_test_X.pt")
X_test_reference_unovis = torch.load("tensors/final_tensors_1703/unovis_reference_test_X.pt")


classifier = ClassificationTrainer(lr=run_config["INIT_LR"], 
                                    batch_size=run_config["BATCH_SIZE"], 
                                    no_epochs=run_config["EPOCHS"], 
                                    model_name=run_config["CLASSIFIER_ARCH"],
                                    X_train=X_train,
                                    y_train=y_train,
                                    X_test=X_validation,
                                    y_test=y_validation,
                                    X_test_reference=X_validation_reference
                                    )
    
# sweep_id = wandb.sweep(sweep_config, project="aum-best-model-final-run")
# wandb.agent(sweep_id, classifier.train, count=1)

classifier.train(run_config=best_config)
results_train_acc, results_train_loss, results_val_acc, results_val_loss = classifier.getRawResults()
val_confusion_matrix = classifier.getBestConfusionMatrix()
classifier.test(X_test=X_test, y_test=y_test, X_reference_test=X_test_reference)

