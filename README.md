This repo contains all the Master Thesis work for the topic "Development of a data driven pipeline for artifact compensation in capacitive ECG signals". 

main branch is outdated, merge it with dev branch if needed.
anything with prefix OLD is outdated.


Structure is as follows:


Preparation:

unovis_reader.py- > reads unovis records, preprocessing and saves as dicts
mit_reader.py -> reads MIT records, preprocessing and saves as dicts
load_data_from_dicts.py -> converts dict data to tensors, applies splitting between training/val/test sets
get_sample_segments.py -> gets example segments from databases
custom_dataset_for_dataloader.py -> implements a Dataset class for DataLoaders
plotter.py -> defines functions to plots, useful for individual runs


Classification:

classification_models.py -> contains model implementations for classification
train.py -> classifier class implementation, used for training in sweeps, for testing use full_pipeline.py


Compensation:

compensation_models.py -> contains model implementations for compensation
train_compensator.py -> compensator class implementation, used for training in sweeps, for testing use full_pipeline.py


Training, Sweeps and Testing:

nxm_quality_assessment.py -> runs hyperparameter sweeps through WandB for compensation and classification
get_best_model.py -> pulls best performing model parameters from WandB and saves the onnx model for visualization
train_best_model.py -> pull best performing model parameters from WandB sweep, re-train and save the trained model as .pt
full_pipeline.py -> used for loading trained and saved models and applying test sets
