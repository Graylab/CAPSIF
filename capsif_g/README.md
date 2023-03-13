# CAPSIF:Graph #


## PREDICTION: ##
* For a directory
```sh
>> python predict_directory.py [pdb_directory: default '../sample_dir' ][model_directory: default './models_DL/cb_model.pth.tar']
```

## REPRODUCTION OF RESULTS ##

### DATASET PREPARATION: ###

#### _You do not need this step for testing. Go to the training step directly_ ####

1. Preprocess the pdbs into the graph readable format:  
```sh
>> ./preprocess.py
```

### TRAINING: ###
First set the model_root_directory in settings.py. This path show the parent directory for CAPSIF codes.

1. Train the model
```sh
>> ./train.py
```
Outputs the best model into "model_DL/capsif_g_model.pth.tar"

### TESTING: ###

1. Test the model on the test dataset from the paper
```sh
>> ./predict_on_testset.py
```
Outputs the Dice score and all TP/FP/FN/TN values
