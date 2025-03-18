import wandb
api = wandb.Api()

sweep = api.sweep("alperencebecik-rwth-aachen-university/aum-classifier-run/4lbajcq2")

# Get best run parameters
best_run = sweep.best_run()
best_parameters = best_run.config
print(best_parameters)