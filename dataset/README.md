# CArbohydrate-Protein interaction Site IdentiFier #


## DATASET ##
**train**: for train dataset  
**val**: for validation dataset  
**test**: for test dataset  
**test_af**: for AlphaFold2 predicted test dataset  

Each directory have two sub-directories:  
**pdbs**: contains pdb files.  
**pdb_npz**: numpy zipped archives of of pdb files. each pdb_npz file contains, precalculated residue features and C_Alpha and C_Beta atom coordinates for each residue. pdb_npz files are used for faster data access for accelerated training.  
The python class **"pdb_to_interaction_file"** from ../data_preparation/pdb_2_interaction_file_coverter.py is used for the making of npz files.  





