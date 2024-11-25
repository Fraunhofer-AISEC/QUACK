# QUACK
Implementation of the Quantum Aligned Centroid Kernel algorithm. 
This project is not maintained. It has been published as part of the following paper:
<https://arxiv.org/abs/2405.00304>


## Installation
All required packages should automatically be installed by creating a docker container
from the provided Dockerfile.

## Usage
- An example of how to use QUACK is given in `train_save_load_model.py`.
- The parameters and results of the models used in the paper can be found in `models/models_test2.paper`
- Due to legacy reasons, the validation set is called "test" and the test set "test2"

## Datasets
- `census_income` (CC BY 4.0 license, <https://archive.ics.uci.edu/dataset/20/census+income>) is provided in `datasets/preprocessed`, for the other datasets see `datasets/datasets.md` 
- If you want to use your own datasets, overwrite the `load_dataset` function in `datasets.py`.
