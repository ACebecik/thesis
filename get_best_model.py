import wandb
api = wandb.Api()

sweep = api.sweep("alperencebecik-rwth-aachen-university/aum-classifier-run-big/dpdrspib")

# Get best run parameters
best_run = sweep.best_run(order="classification_val_acc")
best_parameters = best_run.config
print(best_parameters)