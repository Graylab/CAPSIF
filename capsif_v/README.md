# Carbohydrate-Protein Interaction Site Prediction #

## Prediction ###
**predictor_tool.py**: for the prediction of carbohydrate binding residues on a given protein.  
This is the main carbohydrate binding site prediction utility.   
```sh
>> python predictor_tool.py.py

```

This code will predict the carbohydrate binding residues on input protein.  
If carbohydrate residues are present in input PDB file, this code will calculate the DICE and show other details.

1. Prediction of binding site on a single pdb.

For locally avilable pdb file
```sh
>> python predictor_tool.py <pdbfile with path>

```
For pdb file direct from RCSB
```sh
>> python predictor_tool.py RCSB:<PDB ID>

```
2. Run as a prediction server for multiple pdbs

```sh
>> python predictor_tool.py

```  

#### Example: ####  
![Example](./capsif_v/Figures/prediction_example.png)

Predicted residues on the input protein can be seen on the fly using UCSF Chimera (should be accessible by 'chimera' command in shell. (Linux))

* All test data set at once:  
``>> ./predictor.py``
* Try on one PDB:  
``>> ./predictor_on_pdb2.py``
  * For one prediction  
``>> ./predictor_on_pdb2.py <pdbfile> ``

  * for multiple prediction  
``>> ./predictor_on_pdb2.py``  
[This command loads model and waits for pdbs for prediction]

## DATASET PREPARATION: ##

#### _You do not need this step for testing. Go to the training step directly_ ####

1. Identify Rosetta readable PDB files using:  
``data_preparation/pyrosetta_readable_finding.py``

2. Randomly separate PDB files to Train, Test, and Val types.  
 ####Use np.random.permutation for random indexing and select segments as per your given ratio. or use:####
 ``data_preparation/make_train_and_test_random.py``

3. Make simplified pdb data for train/test/val pdbs using
```data_preparation/pdb_2_interaction_file_converter.py```


## TRAINING: ##
1. modify **train.py** for training and validation directories:  
 [currently set for default data given in dataset directory]

2. Train model using train.py  
[Download pre-trained model for the given data.]  
For GrayLab: get models from louis /mnt/share/carbohydrate/sudhanshu/MODEL_29

### Outputs: ###
**Current best training model:** ./  
**Best model using validation data:** ./models_DL/  
**Stepwise accuracy:** ./Reports/report_xxxx  

## OTHER TOOLS: ##

1. Data analysis:  
	* For unequal-edge-sized first voxelized data:  

	```sh
	>> ./plot_npy_data.py <voxel_protein>  <voxel_protein>
	```

	 * For cube voxels:  

	```sh
	>> plot_npy_data.py <voxel_protein>  <voxel_mask>
	```
	* For real and predicted voxels: (in saved_masks)  

	```sh
	>> plot_npy_data.py <voxel_protein>  <voxel_mask> <voxel_mask_predicted>
	```
