import wandb
import pandas as pd
import matplotlib.pyplot as plt

api = wandb.Api()

sweep = api.sweep("alperencebecik-rwth-aachen-university/ansari-aum-optimization/5c5s7qee")

# Get best run parameters
best_run = sweep.best_run(order="classification_val_acc")

best_parameters = best_run.config
print(best_parameters)

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