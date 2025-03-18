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



if __name__ == "__main__":

    wandb.login()

    # define training hyperparameters as a config dict
    run_config = dict(
        INIT_LR = 5e-4,
        BATCH_SIZE = 4096,
        EPOCHS = 100,
        CHOSEN_DATASET = "um",
        RANDOM_SEED = 31,
        TEST_SIZE = 0.2,
        COMPENSATOR_ARCH = "fcn-dae",
        CLASSIFIER_ARCH = "lstm"
    )
   # wandb.init(project="test_run")
   # wandb.config = run_config

    sweep_config ={
        "method": "random",
        "program": "nxm_quality_assessment.py" 
    } 

    metric ={
       # "name": "compensation_val_loss",
       "name": "classification_val_loss",
        "goal": "minimize"
    } 

    sweep_config["metric"] = metric

    parameters_dict = {
        "CLASSIFIER_ARCH":{
            "values" : ["ansari", "lstm"]  
        } ,
        'COMPENSATOR_ARCH': {
          #  'values': ['fcn-dae', "fcn-dae-skip", "drdnn"]
            "values" :["fcn-dae-skip"] 
                },
        'INIT_LR': {
            "values" :[0.01, 0.0033, 0.001, 0.00033, 0.0001, 0.000033, 0.00001] 
        #    "values" :[0.0001] 
            },
        "BATCH_SIZE":{
            "values" :[1024, 2048, 4096, 8192] 
        #    "values" :[2048] 
        }     
    } 

    sweep_config['parameters'] = parameters_dict

    pprint.pprint(sweep_config)


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
                                        X_test=X_test,
                                        y_test=y_test,
                                        X_test_reference=X_test_reference
                                        )
        
   # classifier.train()
   # results_train_acc, results_train_loss, results_val_acc, results_val_loss = classifier.getRawResults()
   # best_confusion_matrix = classifier.getBestConfusionMatrix()


    compensator = CompensationTrainer(lr=run_config["INIT_LR"],
                                        batch_size=run_config["BATCH_SIZE"],
                                        no_epochs=run_config["EPOCHS"],
                                        model_arch=run_config["COMPENSATOR_ARCH"] ,
                                        X_train=X_train,
                                        y_train=X_train_reference, 
                                        X_test=X_validation,
                                        y_test=X_validation_reference)

   # compensator.train()


   # sweep_id = wandb.sweep(sweep_config, project="augmented-dataset-comparison-run")

   # wandb.agent(sweep_id, compensator.train, count=1)

    sweep_id = wandb.sweep(sweep_config, project="aum-classifier-run")

    wandb.agent(sweep_id, classifier.train, count=10)

    """ 

  #  COMPENSATION PLOTTING

        comp_results_train_loss, comp_results_val_loss = compensator.getRawResults()
                
        plt.plot(comp_results_train_loss, label='Train Loss')
        plt.plot(comp_results_val_loss, label=' Val Loss')

        plt.xlabel('Epochs')
        plt.ylabel("Loss")
        plt.title('Compensation Loss')
        plt.legend()
        plt.savefig(f"plots/newdatasetLoss.png")
        plt.clf()
        
        
        zero_idx_list = np.arange(2000,60000,100)
        max_snaps = 100
        snap_counter = 0
        for i in zero_idx_list:
            if snap_counter == max_snaps:
                break        
            compensator.getRandomSnapshot(random_seed=i)
            snap_counter = snap_counter + 1
        """


    """ 

   # CLASSIFICATION PLOTTING
       
        classification_train_acc, classification_train_loss, classification_val_acc, classification_val_loss = classifier.getRawResults()
   # best_confusion_matrix = classifier.getBestConfusionMatrix()
        plotter = Plotter(dataset=run_config["CHOSEN_DATASET"] , seed=run_config["RANDOM_SEED"] , 
                        lr=run_config["INIT_LR"] , batch_size=run_config["BATCH_SIZE"] )
        plotter.plot_accuracy(results_train_acc,results_val_acc)
        plotter.plot_loss(results_train_loss, results_val_loss)
        plotter.plot_confusion_matrix(best_confusion_matrix)"""




