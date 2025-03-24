import wandb
import pandas as pd
import matplotlib.pyplot as plt
from classification_models import NoiseDetector, LSTMClassifier
from compensation_models import DRDNN, FCN_DAE, FCN_DAE_skip
import torch
import numpy as np


api = wandb.Api()
wandb.init()
sweep = api.sweep("alperencebecik-rwth-aachen-university/ansari-aum-hidden-size-optimization/k3o7p6vs")

# Get best run parameters
best_run = sweep.best_run(order="classification_val_acc")

best_parameters = best_run.config
print(best_parameters)

dummy_input = torch.Tensor (np.ones((1024,1,120)))
classifier_flag = False

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


best_run_values = pd.DataFrame(best_run.history())
print(list(best_run_values.columns.values))

train_df = best_run_values.iloc[::2]
train_df = train_df.reset_index()

val_df = best_run_values.iloc[1::2]
val_df = val_df.reset_index()

plt.plot(val_df["classification_val_acc"], label="Validation Accuracy")
plt.plot(train_df["classification_train_acc"], label = "Training Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(loc="best")
plt.title("Train and Validation Accuracy - CNN")
plt.savefig ("plots/best_runs/ansari_best_run_acc")
plt.clf()

plt.plot(val_df["classification_val_loss"], label="Validation Loss")
plt.plot(train_df["classification_train_loss"], label = "Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(loc="best")
plt.title("Train and Validation Loss - CNN")
plt.savefig ("plots/best_runs/ansari_best_run_loss")