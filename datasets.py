"""
Copyright 2024 Fraunhofer AISEC: Kilian Tscharke
"""


import numpy as np
import torch

def load_dataset(name, n_train_samples, n_test_samples, n_samples_test2=None, log_more_samples=False):
    """ loads preprocessed datasets
    overwrite this function if you want to use a different dataset
    :param name: name of the dataset, available: census_income, cover_type, DoH, emnist, fmnist, KDD, mnist, URL
    :param n_train_samples: number of training samples
    :param n_test_samples: number of validation samples
    :param n_samples_test2: number of test samples
    :param log_more_samples: if True, log(n_train_samples)*n_train_samples samples are loaded for X_train and y_train,
    and n_train_samples samples are loaded for X_train_svm and y_train_svm
    :return: X_train, y_train, X_test, y_test, X_train_svm, y_train_svm, X_test2, y_test2
    X is  a torch tensor of shape (n_samples, n_features)
    y is a torch tensor of shape (n_samples)
    """
    path = "datasets/processed"
    try:
        data = np.load(f"{path}/{name}.npz")
    except:
        raise ValueError(f"Dataset {name} not found.")
    X_test2, y_test2 = None, None
    X_train = torch.tensor(data['X_train'][0:n_train_samples])
    y_train = torch.tensor(data['y_train'][0:n_train_samples])
    if n_samples_test2 is not None:
        X_test2 = torch.tensor(data['X_test'][n_test_samples:n_test_samples + n_samples_test2])
        y_test2 = torch.tensor(data['y_test'][n_test_samples:n_test_samples + n_samples_test2])
    X_test = torch.tensor(data['X_test'][0:n_test_samples])
    y_test = torch.tensor(data['y_test'][0:n_test_samples])
    X_train_svm = torch.tensor(data['X_train'][0:n_train_samples])
    y_train_svm = torch.tensor(data['y_train'][0:n_train_samples])
    return X_train, y_train, X_test, y_test, X_train_svm, y_train_svm, X_test2, y_test2