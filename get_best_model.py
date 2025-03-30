import wandb
import pandas as pd
import matplotlib.pyplot as plt
from classification_models import NoiseDetector, LSTMClassifier
from compensation_models import DRDNN, FCN_DAE, FCN_DAE_skip
import torch
import numpy as np


api = wandb.Api()
wandb.init()
sweep = api.sweep("alperencebecik-rwth-aachen-university/fcn-dae-skip-aum-optimization/4yiejb4j")

# Get best run parameters
best_run = sweep.best_run(order="")

best_parameters = best_run.config
print(best_parameters)

dummy_input = torch.Tensor (np.ones((1024,1,120))).to(device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
classifier_flag = False
compensator_flag = True

if classifier_flag == True:
    if best_parameters["CLASSIFIER_ARCH"] == "ansari":
        model = NoiseDetector(  p_dropout=best_parameters["DROPOUT"],
                                fc_size=best_parameters["ANSARI_HIDDEN_SIZE"])
        # Save the model in the exchangeable ONNX format
        torch.onnx.export(model, dummy_input, "models/ansari_model.onnx")
        wandb.save("ansari_model.onnx")

    else:
        model = LSTMClassifier(config_hidden_size= best_parameters["LSTM_HIDDEN_SIZE"] )
        torch.onnx.export(model, dummy_input, "models/lstm_model.onnx")
        wandb.save("lstm_model.onnx")

if compensator_flag == True:
    if best_parameters["COMPENSATOR_ARCH"] == "fcn-dae":
        model = FCN_DAE(  p_dropout=best_parameters["DROPOUT"])
        # Save the model in the exchangeable ONNX format
        torch.onnx.export(model, dummy_input, "models/fcn_dae_model.onnx")
        wandb.save("fcn_dae_model.onnx")

    elif best_parameters["COMPENSATOR_ARCH"] == "drdnn":
        model = DRDNN(lstm_hidden_size= best_parameters["LSTM_HIDDEN_SIZE"] )
        torch.onnx.export(model, dummy_input, "models/drdnn_model.onnx")
        wandb.save("drdnn_model.onnx")
    
    else:
        model = FCN_DAE_skip().to(device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        model.load_state_dict(torch.load(f"models/fcn-dae-skip.pt"))
        torch.onnx.export(model, dummy_input, "models/fcn_dae_skip_model.onnx")
        wandb.save("fcn_dae_skip_model.onnx")

best_run_values = pd.DataFrame(best_run.history())
print(list(best_run_values.columns.values))

train_df = best_run_values.iloc[::2]
train_df = train_df.reset_index()

val_df = best_run_values.iloc[1::2]
val_df = val_df.reset_index()

"""
plt.plot(val_df["classification_val_acc"], label="Validation Accuracy")
plt.plot(train_df["classification_train_acc"], label = "Training Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(loc="best")
plt.title("Train and Validation Accuracy - LSTM")
plt.savefig ("plots/best_runs/lstm_best_run_acc")
plt.clf()


plt.plot(val_df["classification_val_loss"], label="Validation Loss")
plt.plot(train_df["classification_train_loss"], label = "Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(loc="best")
plt.title("Train and Validation Loss - LSTM")
plt.savefig ("plots/best_runs/lstm_best_run_loss")



"""


"""
# COMPENSATION PLOT

plt.plot(val_df["compensation_val_loss"], label="Validation Loss")
plt.plot(train_df["compensation_train_loss"], label = "Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(loc="best")
plt.title("Train and Validation Loss - DRDNN")
plt.savefig ("plots/best_runs/drdnn_best_run_loss")


"""