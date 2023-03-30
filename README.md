# Carbohydrate-Protein Interaction Site Prediction #

![Architectures](./misc/fig_repo.jpg)

Official repository for [CArbohydrate-Protein Site IdentiFier](https://www.biorxiv.org/content/10.1101/2023.03.14.531382v1.abstract): from the paper
Structure-Based Neural Network Protein-Carbohydrate Interaction Predictions at the Residue Level

Citation:
Structure-Based Neural Network Protein-Carbohydrate Interaction Predictions at the Residue Level
Samuel Canner, Sudhanshu Shanker, Jeffrey Gray

CAPSIF is a deep learning method to determine the carbohydrate-binding residues of proteins given a protein structure. Here we present two CAPSIF models - CAPSIF:Voxel (CAPSIF:V) and CAPSIF:Graph (CAPSIF:G). Both models use convolutions, but different representations of the data. CAPSIF:V uses a voxelized representation and a 3D-[UNet](https://arxiv.org/pdf/1505.04597.pdf) architecture to decipher carbohydrate-binding residues. CAPSIF:G uses a graph representation with an [Equivariant Graph Neural Network](https://arxiv.org/pdf/2102.09844.pdf) architecture. For further details, check out our [paper on bioRxiv](https://www.biorxiv.org/content/10.1101/2023.03.14.531382v1.abstract).

### Quick Setup Guide ###
```
We suggest using a Micromamba/conda environment for the installation.
Steps:
>> micromamba create -n capsif
>> micromamba install python=3.9 -c conda-forge
>> micromamba install pytorch torchvision cudatoolkit=11.5 -c pytorch -c nvidia -c conda-forge
>> micromamba install biopython pandas colorama scikit-learn matplotlib tqdm py3dmol -c conda-forge
To install pyrosetta, create a ~/.mambarc (.condarc) file with the following content:
--------------------------------------------------------
channels:
- https://USERNAME:PASSWORD@conda.rosettacommons.org
- defaults
--------------------------------------------------------
>> micromamba install pyrosetta

```
or  


get the capsif.yml file from mamba_environment directory  
set the usename and password for pyrosetta access.  
and run:
```sh
>> micromamba create -f capsif.yml -no_rc
```

##### Download Weights #####
First run the commands
```
mkdir capsif_v/models_DL/
mkdir capsif_g/models_DL/
```

The weights of each model are stored on our remote server `data.graylab.jhu.edu/CAPSIF/`

Download `my_checkpoint_best_36_2A_CACB_vector_coord_I_clean_data.pth.tar` to `CAPSIF/capsif_v/models_DL/`

Download `cb_model.pth.tar` to `CAPSIF/capsif_g/models_DL/`

### SYSTEM REQUIREMENTS: ###

```
Python3 (3.9)  
PyRosetta4 2023.06+release.27ba97a  py39_0  
biopython==1.81
colorama==0.4.6
IO==0.0.1  
matplotlib==3.7.0  
numpy==1.24.2  
pandas==1.5.3  
pytorch==1.13.1 py3.9_cuda11.7_cudnn8.5.0_0
tqdm==4.64.0  
py3Dmol==1.8.0
```

## PREDICTION: ##
* For a single PDB you can use the Jupyter Notebook
``./sample_notebook.ipynb``
* For prediction of multiple PDBs in a directory, we primarily suggest use of
``./predict_directory.py`` or ``./notebook_predict_directory.ipynb``

* Usage of ``predict_directory.py``
```
>> python ./predict_directory.py --dir [working_directory/default: 'sample_dir/']
    --v_model [CAPSIF:V Model/default:"./capsif_v/models_DL/my_checkpoint_best_36_2A_CACB_vector_coord_I_clean_data.pth.tar"]
    --g_model [CAPSIF:G Model/default:"./capsif_g/models_DL/cb_model.pth.tar"]
    --out [output_directory/defalt: 'sample_dir']
    --make_pdb [default: True]
```

Returns: `[output_directory]/capsif_predictions.txt` with a list of binding residues for each protein and model
    and `[output_directory]/*.pdb` with the pdbs with the BFactor identifying the binding Residues

<ins>Current settings for B Factor visualization</ins>

``BFactor =  0.0`` : Nonbinder

``BFactor = 40.0`` : CAPSIF:G Predicted Binder

``BFactor = 59.9`` : CAPSIF:V Predicted Binder

``BFactor = 99.9`` : CAPSIF:V and CAPSIF:G Predicted Binder

#### For conventional command line predictions for each model ####
* CAPSIF:V - [`./capsif_v/README.md`](./capsif_v/README.md) for usage of the chimera-integrated script `predictor_tool.py`

* CAPSIF:G - [`./capsif_g/README.md`](./capsif_g/README.md) for usage of `predict_directory.py`

## PAPER REPRODUCTION ##

### [DATASET INFORMATION](./misc/dataset.csv) ###

### DATASET PREPARATION: ###

#### _You do not need this step for testing. Go to the training step directly_ ####

1. Identify Rosetta readable PDB files using:  
``data_preparation/pyrosetta_readable_finding.py``

2. Randomly separate PDB files to Train, Test, and Val types.  
 #### Use np.random.permutation for random indexing and select segments as per your given ratio. or use: ####
 ``data_preparation/make_train_and_test_random.py``

3. Make simplified pdb data files for faster access during train/test/val using
```data_preparation/pdb_2_interaction_file_converter.py```

### TRAINING and TESTING ###

For each model's reproduction after running data_preparation steps, please refer to each directory's README

### [CAPSIF_V](./capsif_g/README.md) ###
### [CAPSIF:G](./capsif_g/README.md) ###
