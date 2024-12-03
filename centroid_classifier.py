"""
Copyright 2024 Fraunhofer AISEC: Kilian Tscharke
"""


import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.svm import SVC

import datasets as ds
from statevector_simulator import StateVecSimTorch


class CentroidClassifier:
    def __init__(self, n_layers, n_qubits, init_weights_scale,
                 dataset, n_samples_train=200, n_samples_test=100,
                 reg_param_ka=0.0, reg_param_co=0.0, epochs=10, epochs_ka=10, epochs_co=10,
                 lr_ka=0.1, lr_co=0.1, decay_rate=0.9, gpu=False, silent=False, seed=42,
                 model=None, n_samples_test2=None, **kwargs):
        """
        :param n_layers: number of layers in the quantum circuit
        :param n_qubits: number of qubits in the quantum circuit
        :param init_weights_scale: scale of the weights and biases during initalization of the classifier
        :param dataset: name of the dataset, see load_dataset in datasets.py
        :param n_samples_train: number of training samples
        :param n_samples_test: number of validation samples
        :param reg_param_ka: regularization parameter for kernel alignment
        :param reg_param_co: regularization parameter for centroid optimization
        :param epochs: number of epochs for training the classifier
        :param epochs_ka: number of epochs for the kernel alignment optimization
        :param epochs_co: number of epochs for the centroid optimization
        :param lr_ka: learning rate for kernel alignment optimization
        :param lr_co: learning rate for centroid optimization
        :param decay_rate: decay rate for the learning rate
        :param gpu: if True, the model is trained on the GPU
        :param silent: if True, no output is printed
        :param seed: random seed
        :param model: id of the model to be loaded
        :param n_samples_test2: number of test samples
        """
        self.n_layers = n_layers
        self.n_qubits = n_qubits
        X_train_cpu, y_train_cpu, X_test_cpu, y_test_cpu, X_train_svm, y_train_svm, X_test2, y_test2 = \
            ds.load_dataset(dataset, n_samples_train, n_samples_test, n_samples_test2)

        simulator = StateVecSimTorch(n_qubits=n_qubits, n_layers=n_layers, init_weights_scale=init_weights_scale,
                                     gpu=gpu, seed=seed)

        if gpu:
            # make ready for gpu
            try:
                device = torch.device("cuda")
            except:
                print("GPU not available, continue on CPU")
                device = torch.device("cpu")
        else:
            device = torch.device("cpu")
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        simulator = simulator.to(device)
        X_train = X_train_cpu.to(device)
        X_test = X_test_cpu.to(device)
        y_train = y_train_cpu.to(device)
        y_test = y_test_cpu.to(device)

        self.gpu = gpu
        self.dataset = dataset
        self.n_samples_train = len(X_train)
        self.n_samples_test = len(X_test)
        self.n_samples_test2 = n_samples_test2
        self.num_queries = 0
        self.reg_param_ka = reg_param_ka
        self.reg_param_co = reg_param_co
        self.init_weights_scale = init_weights_scale
        self.epochs = epochs
        self.epochs_ka = epochs_ka
        self.epochs_co = epochs_co
        self.lr_ka = lr_ka
        self.lr_co = lr_co
        self.decay_rate = decay_rate
        self.seed = seed
        self.silent = silent
        self.train_loss = None
        self.test_loss = None
        self.train_acc = None
        self.test_acc = None
        self.train_auc = None
        self.test_auc = None
        self.train_f1 = None
        self.test_f1 = None
        self.svm_rbf_train_acc = None
        self.svm_rbf_test_acc = None
        self.svm_rbf_train_auc = None
        self.svm_rbf_test_auc = None
        self.svm_rbf_train_f1 = None
        self.svm_rbf_test_f1 = None
        self.qsvm_train_acc = None
        self.qsvm_test_acc = None
        self.qsvm_train_auc = None
        self.qsvm_test_auc = None
        self.qsvm_train_f1 = None
        self.qsvm_test_f1 = None
        self.num_svs_rbf = None
        self.num_svs_qsvm = None
        self.rbf_centroid_acc = None
        self.rbf_centroid_auc = None
        self.rbf_centroid_f1 = None
        self.test_acc2 = None
        self.test_loss2 = None
        self.roc_auc_test2 = None
        self.f1_test2 = None
        self.svm_rbf_test2_acc = None
        self.svm_rbf_test2_auc = None
        self.svm_rbf_test2_f1 = None
        self.rbf_centroid_acc2 = None
        self.rbf_centroid_auc2 = None
        self.rbf_centroid_f12 = None
        self.qsvm_test2_acc = None
        self.qsvm_test2_auc = None
        self.qsvm_test2_f1 = None
        self.losses = []

        self.X_train, self.y_train, self.X_test, self.y_test = X_train, y_train, X_test, y_test
        self.X_test2, self.y_test2 = X_test2, y_test2
        self.X_train_cpu, self.y_train_cpu, self.X_test_cpu, self.y_test_cpu = X_train_cpu, y_train_cpu, X_test_cpu, y_test_cpu
        self.X_train_svm, self.y_train_svm, = X_train_svm, y_train_svm
        self.centroids = self.get_init_centroids(X_train, y_train).to(device)
        self.centroid_classes = [1, -1]
        self.model = simulator
        if model:
            self.model.load_state_dict(torch.load(f"models/models/{model}.pt", map_location=torch.device('cpu'), weights_only=True))
            self.centroids = torch.load(f"models/centroids/{model}.pt", map_location=torch.device('cpu'), weights_only=True)

    def get_init_centroids(self, X_train, y_train):
        c1 = torch.mean(X_train[y_train == 1], dim=0).view(1, X_train.shape[1])
        c2 = torch.mean(X_train[y_train == -1], dim=0).view(1, X_train.shape[1])
        centroids = torch.cat([c1, c2], dim=0)
        return centroids

    def get_kernel_quadratic(self, state):
        """
        :return: kernel matrix of shape (n_samples, n_samples)
        """
        return torch.abs(state.conj() @ state.T) ** 2

    def get_kernel_rectangular(self, state_rows, state_cols):
        """
        :return: kernel matrix of shape(n_samples, n_samples)
        """
        return torch.abs(state_rows.conj() @ state_cols.T) ** 2

    def get_kernel_alignment(self, kernel1, kernel2):
        """
        get kernel-target alignment as in
        http://arxiv.org/abs/2105.02276 eq 26
         kernel alignment is -1 if vectors orthogonal, 1 if vectors identical
        :param kernel1: kernel matrix of shape (n_samples, n_samples)
        k(xi,xj) should be 1 if xi and xj have the same label, 0 if xi and xj have different labels
        :param kernel2: ideal kernel matrix of shape (n_samples, n_samples).
        k(xi,xj) = 1 if xi and xj have the same label, -1 if xi and xj have different labels
        if kernel2 is for a single centroid, kernel2 is y (or -y if centroid_class == -1)
        :return: kernel alignment of shape (n_samples, n_samples)
        """
        kernel1 = kernel1.type(torch.complex64)
        kernel2 = kernel2.type(torch.complex64)
        fip1 = torch.sum(torch.mul(kernel1, kernel1))
        fip2 = torch.sum(torch.mul(kernel2, kernel2))
        fip12 = torch.sum(torch.mul(kernel1, kernel2))
        kernel_alignment = fip12 / (torch.sqrt(fip1) * torch.sqrt(fip2))

        return kernel_alignment

    def loss_rectangular(self, kernel, y):
        """
        :param kernel: kernel matrix of shape (n_samples, n_centroids).
        column 0: class 1, column 1: class -1
        :param y: true labels of shape (n_samples, )
        :return: loss
        """
        kernel_ideal = torch.ones_like(kernel)
        kernel_ideal[y == -1, 0] = -1
        kernel_ideal[y == 1, 1] = -1
        ka = self.get_kernel_alignment(kernel, kernel_ideal)

        loss = torch.real(- ka + 1)
        return loss

    def loss_single_centroid_ka(self, kernel, y, centroid_class):
        if centroid_class == -1:
            y = -y
        ka = self.get_kernel_alignment(kernel, y.view(-1, 1))
        weights = self.model.weights
        regularization = torch.sum(weights ** 2)
        loss = torch.real(- ka + 1) + self.reg_param_ka * regularization
        return loss

    def loss_single_centroid_co(self, kernel, y, centroid_class):
        if centroid_class == -1:
            y = -y
        ka = self.get_kernel_alignment(kernel, y.view(-1, 1))
        curr_centroid = self.centroids[self.centroid_classes.index(centroid_class)].cpu()
        out_of_box = curr_centroid - torch.tensor([1. for _ in range(len(curr_centroid))])
        regularization1 = out_of_box[out_of_box > 0].sum()
        regularization2 = curr_centroid[curr_centroid < 0].sum() * (-1)
        regularization = regularization1 + regularization2
        loss = torch.real(- ka + 1) + self.reg_param_co * regularization
        return loss

    def loss_quadratic(self, kernel, y1, y2=None):
        """
        loss is how strongly kernels are aligned: loss = 0 if kernels are identical, loss = 2 if kernels are orthogonal
        """
        if y2 is None:
            y2 = y1
        ideal_kernel = torch.outer(y1, y2)
        kernel_alignment = self.get_kernel_alignment(kernel, ideal_kernel)

        # kernel alignment is -1 if vectors orthogonal, 1 if vectors identical
        loss = torch.real(- kernel_alignment + 1)
        return loss

    def stepwise_centroid_kernel_alignment(self, epochs, lr, centroid_class, decay_rate, silent, twostep_epoch=None):
        opt = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        opt_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt, gamma=decay_rate)
        results = []
        centroid = self.centroids[self.centroid_classes.index(centroid_class)].view(1, -1)

        for epoch in range(epochs):
            opt.zero_grad()
            states_train = self.model(self.X_train)
            state_centroid = self.model(centroid)
            train_kernel = self.get_kernel_rectangular(states_train, state_centroid)
            loss = self.loss_single_centroid_ka(train_kernel, self.y_train, centroid_class)
            loss.backward()
            states_test = self.model(self.X_test)
            test_kernel = self.get_kernel_rectangular(states_test, state_centroid)
            test_loss = self.loss_single_centroid_ka(test_kernel, self.y_test, centroid_class).item()
            opt.step()
            opt_scheduler.step()

            res = [epoch + 1, loss.item()]
            results.append(res)
            if not silent:
                print("KA Epoch: {:3d} | Loss train: {:.6f}".format(*res))
            losses_dic = {"epoch": twostep_epoch, "ka_epoch": epoch, "co_epoch": None, "train_loss": loss.item(),
                          "test_loss": test_loss, "train_acc": None, "test_acc": None, "num_svs_rbf": None,
                          "num_svs_qsvm": None, "loss_type": "ka_single_centroid"}
            self.losses.append(losses_dic)
        return results

    def stepwise_centroid_optimization(self, epochs, lr, centroid_class, decay_rate, silent, twostep_epoch=None):
        centroid = self.centroids[self.centroid_classes.index(centroid_class)].view(1, -1).\
            clone().detach().requires_grad_(True)
        opt = torch.optim.SGD([centroid], lr=lr, momentum=0.9)
        opt_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt, gamma=decay_rate)
        results = []
        for epoch in range(epochs):
            opt.zero_grad()
            states_train = self.model(self.X_train)
            state_centroid = self.model(centroid)
            train_kernel = self.get_kernel_rectangular(states_train, state_centroid)
            loss = self.loss_single_centroid_co(train_kernel, self.y_train, centroid_class)
            loss.backward()
            opt.step()
            opt_scheduler.step()
            states_test = self.model(self.X_test)
            test_kernel = self.get_kernel_rectangular(states_test, state_centroid)
            test_loss = self.loss_single_centroid_co(test_kernel, self.y_test, centroid_class).item()

            res = [epoch + 1, loss.item()]
            results.append(res)
            if not silent:
                print("CO Epoch: {:3d} | Loss train: {:.6f}".format(*res))
            losses_dic = {"epoch": twostep_epoch, "ka_epoch": None, "co_epoch": epoch, "train_loss": loss.item(),
                          "test_loss": test_loss, "train_acc": None, "test_acc": None, "num_svs_rbf": None,
                          "num_svs_qsvm": None, "loss_type": "co_single_centroid"}
            self.losses.append(losses_dic)
        self.centroids[self.centroid_classes.index(centroid_class)] = centroid.clone().detach()
        return results

    def rbf_kernel(self, X, C, sigma=1.0):
        # Calculate the squared Euclidean distance between each pair of points
        X_sq_norms = np.sum(X ** 2, axis=1).reshape(-1, 1)  # Shape (n_samples, 1)
        C_sq_norms = np.sum(C ** 2, axis=1)  # Shape (n_centroids,)
        cross_term = np.dot(X, C.T)  # Shape (n_samples, n_centroids)
        sq_dists = X_sq_norms + C_sq_norms - 2 * cross_term  # Broadcasting to get pairwise distances

        # Apply the RBF kernel function
        K = np.exp(-sq_dists / (2 * sigma ** 2))
        return K

    def eval_model(self, epoch=None):
        # evaluate on train samples
        states_train = self.model(self.X_train)
        states_centroids = self.model(self.centroids)
        kernel_train = self.get_kernel_rectangular(states_train, states_centroids).cpu()
        y_pred_train = torch.where(kernel_train[:, 0] > kernel_train[:, 1], 1, -1).cpu()
        train_acc = torch.sum(y_pred_train == self.y_train_cpu) / self.n_samples_train
        self.train_acc = train_acc.item()
        if self.train_acc < 0.5:
            y_pred_train = -y_pred_train
        self.train_loss = self.loss_rectangular(kernel_train, self.y_train).item()
        self.roc_auc_train = roc_auc_score(self.y_train_cpu,
                                           kernel_train[:, 0].detach().numpy() - kernel_train[:, 1].detach().numpy())
        self.f1_train = f1_score(self.y_train_cpu, y_pred_train)

        # evaluate on test samples
        states_test = self.model(self.X_test)
        kernel_test = self.get_kernel_rectangular(states_test, states_centroids).cpu()
        y_pred_test = torch.where(kernel_test[:, 0] > kernel_test[:, 1], 1, -1).cpu()
        test_acc = torch.sum(y_pred_test == self.y_test_cpu) / self.n_samples_test
        self.test_acc = test_acc.item()
        if self.test_acc < 0.5:
            y_pred_test = -y_pred_test
        self.test_loss = self.loss_rectangular(kernel_test, self.y_test).item()
        self.roc_auc_test = roc_auc_score(self.y_test_cpu,
                                          kernel_test[:, 0].detach().numpy() - kernel_test[:, 1].detach().numpy())
        self.f1_test = f1_score(self.y_test_cpu, y_pred_test)

        # if test2 use final test set
        if self.n_samples_test2 is not None:
            states_test2 = self.model(self.X_test2)
            kernel_test2 = self.get_kernel_rectangular(states_test2, states_centroids)
            y_pred_test2 = torch.where(kernel_test2[:, 0] > kernel_test2[:, 1], 1, -1)
            test_acc2 = torch.sum(y_pred_test2 == self.y_test2) / self.n_samples_test2
            self.test_acc2 = test_acc2.item()
            if self.test_acc2 < 0.5:
                y_pred_test2 = -y_pred_test2
            self.test_loss2 = self.loss_rectangular(kernel_test2, self.y_test2).item()
            self.roc_auc_test2 = roc_auc_score(self.y_test2, kernel_test2[:, 0].detach().numpy() - kernel_test2[:,
                                                                                                   1].detach().numpy())
            self.f1_test2 = f1_score(self.y_test2, y_pred_test2)
            print(
                f"Test2 accuracy: {self.test_acc2:.3f} | Test2 AUC: {self.roc_auc_test2:.3f} | Test2 F1: {self.f1_test2:.3f}")

        # compare to SVM
        if self.svm_rbf_train_acc is None:
            rbf_svm = SVC(kernel='rbf', probability=True)
            rbf_svm.fit(self.X_train_svm, self.y_train_svm)
            svm_rbf_train_pred = rbf_svm.predict(self.X_train_svm)
            svm_rbf_train_pred_prob = rbf_svm.predict_proba(self.X_train_svm)
            svm_rbf_test_pred = rbf_svm.predict(self.X_test_cpu)
            svm_rbf_test_pred_prob = rbf_svm.predict_proba(self.X_test_cpu)
            self.svm_rbf_train_acc = accuracy_score(self.y_train_svm, svm_rbf_train_pred)
            self.svm_rbf_test_acc = accuracy_score(self.y_test_cpu, svm_rbf_test_pred)
            self.svm_rbf_train_auc = roc_auc_score(self.y_train_svm, svm_rbf_train_pred_prob[:, 1])
            self.svm_rbf_test_auc = roc_auc_score(self.y_test_cpu, svm_rbf_test_pred_prob[:, 1])
            self.svm_rbf_train_f1 = f1_score(self.y_train_svm, svm_rbf_train_pred)
            self.svm_rbf_test_f1 = f1_score(self.y_test_cpu, svm_rbf_test_pred)
            # use this bc qsvm.support_vectors_ does not work for precomputed kernel
            self.num_svs_rbf = sum(rbf_svm.n_support_)

            if self.n_samples_test2 is not None:
                svm_rbf_test2_pred = rbf_svm.predict(self.X_test2)
                svm_rbf_test2_pred_prob = rbf_svm.predict_proba(self.X_test2)
                self.svm_rbf_test2_acc = accuracy_score(self.y_test2, svm_rbf_test2_pred)
                self.svm_rbf_test2_auc = roc_auc_score(self.y_test2, svm_rbf_test2_pred_prob[:, 1])
                self.svm_rbf_test2_f1 = f1_score(self.y_test2, svm_rbf_test2_pred)
                print(f"SVM RBF: Test2 accuracy: {self.svm_rbf_test2_acc:.3f} | Test2 AUC: {self.svm_rbf_test2_auc:.3f} | Test2 F1: {self.svm_rbf_test2_f1:.3f}")

            # rbf centroid classifier
            # no training => only test samples matter
            # sigma from https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
            # kernel matrix from https://www.askpython.com/python-modules/numpy/gaussian-kernel-matrix-numpy
            sigma = 1 / (self.X_test_cpu.shape[1] * self.X_test_cpu.var())
            dataset_means = self.get_init_centroids(self.X_train_cpu, self.y_train_cpu).numpy()
            rbf_centroid_kernel = self.rbf_kernel(self.X_test_cpu.detach().numpy(), dataset_means, sigma)
            rbf_centroid_preds = np.where(rbf_centroid_kernel[:, 0] > rbf_centroid_kernel[:, 1], 1, -1)
            self.rbf_centroid_acc = accuracy_score(self.y_test_cpu, rbf_centroid_preds)
            self.rbf_centroid_auc = roc_auc_score(self.y_test_cpu,
                                                  rbf_centroid_kernel[:, 0] - rbf_centroid_kernel[:, 1])
            self.rbf_centroid_f1 = f1_score(self.y_test_cpu, rbf_centroid_preds)

            if self.n_samples_test2 is not None:
                dataset_means = self.get_init_centroids(self.X_train_cpu, self.y_train_cpu).numpy()
                rbf_centroid_kernel2 = self.rbf_kernel(self.X_test2.numpy(), dataset_means, sigma)
                rbf_centroid_preds2 = np.where(rbf_centroid_kernel2[:, 0] > rbf_centroid_kernel2[:, 1], 1, -1)
                self.rbf_centroid_acc2 = accuracy_score(self.y_test2, rbf_centroid_preds2)
                self.rbf_centroid_auc2 = roc_auc_score(self.y_test2,
                                                       rbf_centroid_kernel2[:, 0] - rbf_centroid_kernel2[:, 1])
                self.rbf_centroid_f12 = f1_score(self.y_test2, rbf_centroid_preds2)
                print(
                    f"RBF Centroid: Test2 accuracy: {self.rbf_centroid_acc2:.3f} | Test2 AUC: {self.rbf_centroid_auc2:.3f} | Test2 F1: {self.rbf_centroid_f12:.3f}")

        print(f"SVM RBF: Train accuracy: {self.svm_rbf_train_acc:.3f} | Test Accuracy: {self.svm_rbf_test_acc:.3f} | "
              f"Number of SVs: {self.num_svs_rbf}"
              f"| Train AUC: {self.svm_rbf_train_auc:.3f} | Test AUC: {self.svm_rbf_test_auc:.3f} | "
              f"Train F1: {self.svm_rbf_train_f1:.3f} | Test F1: {self.svm_rbf_test_f1:.3f}")
        print(f"RBF Centroid: Test accuracy: {self.rbf_centroid_acc:.3f} | Test AUC: {self.rbf_centroid_auc:.3f} | "
              f"Test F1: {self.rbf_centroid_f1:.3f}")

        # QSVM with quadratic kernel
        if self.gpu:
            states_train_svm = self.model(self.X_train_svm).cpu()
            states_test_cpu = states_test.cpu()
        else:
            states_train_svm = states_train
            states_test_cpu = states_test
        qsvm = SVC(kernel='precomputed', probability=True)
        qsvm_train_kernel = self.get_kernel_quadratic(states_train_svm).detach().numpy()
        qsvm_test_kernel = self.get_kernel_rectangular(states_test_cpu, states_train_svm).detach().numpy()
        qsvm.fit(qsvm_train_kernel, self.y_train_svm)
        qsvm_train_pred = qsvm.predict(qsvm_train_kernel)
        qsvm_train_pred_prob = qsvm.predict_proba(qsvm_train_kernel)
        qsvm_test_pred = qsvm.predict(qsvm_test_kernel)
        qsvm_test_pred_prob = qsvm.predict_proba(qsvm_test_kernel)
        self.qsvm_train_acc = accuracy_score(self.y_train_svm, qsvm_train_pred)
        self.qsvm_test_acc = accuracy_score(self.y_test_cpu, qsvm_test_pred)
        self.qsvm_train_auc = roc_auc_score(self.y_train_svm, qsvm_train_pred_prob[:, 1])
        self.qsvm_test_auc = roc_auc_score(self.y_test_cpu, qsvm_test_pred_prob[:, 1])
        self.qsvm_train_f1 = f1_score(self.y_train_svm, qsvm_train_pred)
        self.qsvm_test_f1 = f1_score(self.y_test_cpu, qsvm_test_pred)
        self.num_svs_qsvm = sum(qsvm.n_support_)
        print(
            f"QSVM:    Train accuracy: {self.qsvm_train_acc:.3f} | Test Accuracy: {self.qsvm_test_acc:.3f} | Number of SVs: {self.num_svs_qsvm} "
            f"| Train AUC: {self.qsvm_train_auc:.3f} | Test AUC: {self.qsvm_test_auc:.3f}"
            f"| Train F1: {self.qsvm_train_f1:.3f} | Test F1: {self.qsvm_test_f1:.3f}")
        if self.n_samples_test2 is not None:
            qsvm_test2_kernel = self.get_kernel_rectangular(self.model(self.X_test2).cpu(),
                                                            states_train_svm).detach().numpy()
            qsvm_test2_pred = qsvm.predict(qsvm_test2_kernel)
            qsvm_test2_pred_prob = qsvm.predict_proba(qsvm_test2_kernel)
            self.qsvm_test2_acc = accuracy_score(self.y_test2, qsvm_test2_pred)
            self.qsvm_test2_auc = roc_auc_score(self.y_test2, qsvm_test2_pred_prob[:, 1])
            self.qsvm_test2_f1 = f1_score(self.y_test2, qsvm_test2_pred)
            print(
                f"QSVM: Test2 accuracy: {self.qsvm_test2_acc:.3f} | Test2 AUC: {self.qsvm_test2_auc:.3f} | Test2 F1: {self.qsvm_test2_f1:.3f}")

        if epoch is not None:
            res = [epoch + 1, self.train_loss, self.test_loss, self.train_acc, self.test_acc,
                   self.roc_auc_train, self.roc_auc_test, self.f1_train, self.f1_test]
            print(
                "Epoch {:d} | Cost: {:.3f} | Cost Test: {:.3f} | Train accuracy: {:.3f} | "
                "Test Accuracy: {:.3f} | AUC Train: {:.3f} | AUC Test: {:.3f} |"
                " F1 Train: {:.3f} | F1 Test: {:.3f}".format(
                    *res
                )
            )
        else:
            res = [self.train_loss, self.test_loss, self.train_acc, self.test_acc,
                   self.roc_auc_train, self.roc_auc_test, self.f1_train, self.f1_test]
            print(
                "Initial/Final Results | Cost: {:.3f} | Cost Test: {:.3f} | Train accuracy: {:.3f} | "
                "Test Accuracy: {:.3f} | AUC Train: {:.3f} | AUC Test: {:.3f} |"
                " F1 Train: {:.3f} | F1 Test: {:.3f}".format(
                    *res
                )
            )
        if len(self.centroids[0]) < 10:
            print("Centroids: ", self.centroids[0], self.centroids[1])

        losses_dic = {"epoch": epoch, "ka_epoch": None, "co_epoch": None, "train_loss": self.train_loss,
                      "test_loss": self.test_loss, "train_acc": self.train_acc,
                      "test_acc": self.test_acc, "num_svs_rbf": self.num_svs_rbf,
                      "num_svs_qsvm": self.num_svs_qsvm, "loss_type": "eval_two_centroids",
                      "roc_auc_train": self.roc_auc_train, "roc_auc_test": self.roc_auc_test}
        self.losses.append(losses_dic)

        return res

    def save_model(self, modelname=None):
        """
        if modelname is None, saves params and state of current model.
        if modelname is the model id, the function assumes that the model was evaluated on test set and saves the results
        in models_test2.csv
        :param modelname: model id of the model that was evaluated on test set
        :return: model id of the saved model
        """
        if modelname is None:
            now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            torch.save(self.model.state_dict(), f"models/models/{now}.pt")
            torch.save(self.centroids, f"models/centroids/{now}.pt")
            # save losses
            loss_df = pd.DataFrame(self.losses)
            loss_df.to_csv(f"models/losses/{now}.csv", sep=";", index=False, header=True, decimal=",")
        else:
            now = modelname

        param_dic = {
            "model": now,
            "n_layers": self.n_layers,
            "n_qubits": self.n_qubits,
            "n_samples_train": self.n_samples_train,
            "n_samples_test": self.n_samples_test,
            "epochs": self.epochs,
            "epochs_ka": self.epochs_ka,
            "epochs_co": self.epochs_co,
            "lr_ka": self.lr_ka,
            "lr_co": self.lr_co,
            "decay_rate": self.decay_rate,
            "reg_param_ka": self.reg_param_ka,
            "reg_param_co": self.reg_param_co,
            "dataset": self.dataset,
            "train_loss": self.train_loss,
            "test_loss": self.test_loss,
            "train_acc": self.train_acc,
            "test_acc": self.test_acc,
            "num_svs_rbf": self.num_svs_rbf,
            "num_svs_qsvm": self.num_svs_qsvm,
            "init_weights_scale": self.init_weights_scale,
            "train_auc": self.roc_auc_train,
            "test_auc": self.roc_auc_test,
            "train_f1": self.f1_train,
            "test_f1": self.f1_test,
            "svm_rbf_train_acc": self.svm_rbf_train_acc,
            "svm_rbf_test_acc": self.svm_rbf_test_acc,
            "svm_rbf_train_auc": self.svm_rbf_train_auc,
            "svm_rbf_test_auc": self.svm_rbf_test_auc,
            "svm_rbf_train_f1": self.svm_rbf_train_f1,
            "svm_rbf_test_f1": self.svm_rbf_test_f1,
            "qsvm_train_acc": self.qsvm_train_acc,
            "qsvm_test_acc": self.qsvm_test_acc,
            "qsvm_train_auc": self.qsvm_train_auc,
            "qsvm_test_auc": self.qsvm_test_auc,
            "qsvm_train_f1": self.qsvm_train_f1,
            "qsvm_test_f1": self.qsvm_test_f1,
            "rbf_centroid_acc": self.rbf_centroid_acc,
            "rbf_centroid_auc": self.rbf_centroid_auc,
            "rbf_centroid_f1": self.rbf_centroid_f1,
            "test_acc2": self.test_acc2,
            "test_loss2": self.test_loss2,
            "roc_auc_test2": self.roc_auc_test2,
            "f1_test2": self.f1_test2,
            "svm_rbf_test2_acc": self.svm_rbf_test2_acc,
            "svm_rbf_test2_auc": self.svm_rbf_test2_auc,
            "svm_rbf_test2_f1": self.svm_rbf_test2_f1,
            "rbf_centroid_acc2": self.rbf_centroid_acc2,
            "rbf_centroid_auc2": self.rbf_centroid_auc2,
            "rbf_centroid_f12": self.rbf_centroid_f12,
            "qsvm_test2_acc": self.qsvm_test2_acc,
            "qsvm_test2_auc": self.qsvm_test2_auc,
            "qsvm_test2_f1": self.qsvm_test2_f1,
            "n_samples_test2": self.n_samples_test2,
            "seed": self.seed,
        }
        if modelname is None:
            # save hyper parameters and final results as new row
            df = pd.DataFrame.from_dict([param_dic])
            df.to_csv(f"models/models.csv", mode='a', sep=";", index=False, header=False, decimal=",")
            # if file doesn't exist yet use command below instead
            #df.to_csv(f"models/models.csv", mode='w', sep=";", index=False, header=True, decimal=",")
        else:
            # save hyper parameters and final results as new row if it doesn't exist yet
            df = pd.read_csv(f"models/models_test2.csv", sep=";", decimal=",")
            if not df["model"].str.contains(modelname).any():
                df_newrow = pd.DataFrame.from_dict([param_dic])
                df_newrow.to_csv(f"models/models_test2.csv", mode='a', sep=";", index=False, header=False, decimal=",")
            else:
                # update row
                df.loc[df["model"] == modelname, param_dic.keys()] = param_dic.values()
                df.to_csv(f"models/models_test2.csv", mode='w', sep=";", index=False, header=True, decimal=",")

        return now

    def train(self):
        results = []
        # initial results
        res = self.eval_model()
        centroid_class = self.centroid_classes[0]
        lr_ka_epoch = self.lr_ka
        lr_co_epoch = self.lr_co
        for epoch in range(self.epochs):
            self.stepwise_centroid_kernel_alignment(
                self.epochs_ka, lr_ka_epoch, centroid_class, self.decay_rate, self.silent, epoch)
            centroid_class *= -1
            self.stepwise_centroid_optimization(
                self.epochs_co, lr_co_epoch, centroid_class, self.decay_rate, self.silent, epoch)

            res = self.eval_model(epoch)
            results.append(res)

            lr_ka_epoch = self.lr_ka * self.decay_rate ** (epoch // 2 + 1)
            lr_co_epoch = self.lr_co * self.decay_rate ** (epoch // 2 + 1)

    def plot_decision_surface(self):
        """plot difference of fidelity with centroid 1 and centroid 2
        for a grid of points in 2D space"""
        n_samples_feature = 50
        x1 = np.linspace(0, 1, n_samples_feature)
        x2 = np.linspace(0, 1, n_samples_feature)
        x1, x2 = np.meshgrid(x1, x2)
        X = torch.tensor(np.concatenate((x1.reshape(-1, 1), x2.reshape(-1, 1)), axis=1))
        states = self.model(X)
        states_centroids = self.model(self.centroids)
        kernel = self.get_kernel_rectangular(states, states_centroids).detach().numpy()
        diff = kernel[:, 0] - kernel[:, 1]
        diff = diff.reshape(n_samples_feature, n_samples_feature)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # plot 0 plane
        ax.plot_surface(x1, x2, np.zeros_like(diff), color='gray', alpha=0.5)

        # plot decision surface
        surf = ax.plot_surface(x1, x2, diff, cmap='bwr', vmin=-1, vmax=1)
        fig.colorbar(surf, shrink=0.5, aspect=5)

        # plot train data points
        ax.scatter(self.X_train[self.y_train == 1, 0], self.X_train[self.y_train == 1, 1], 1, c='r')
        ax.scatter(self.X_train[self.y_train == -1, 0], self.X_train[self.y_train == -1, 1], -1, c='b')
        plt.show()


def load_model(model_id, n_samples_test2=None):
    """
    load model from models.csv
    :param model_id: id of the model to be loaded
    :param n_samples_test2: set this to the number of test samples if model will be evaluated on test set
    :return:
    """
    df = pd.read_csv("models/models.csv", sep=";", decimal=",")
    df.drop(df.filter(regex="Unname"), axis=1, inplace=True)
    params = df[df["model"] == model_id].to_dict(orient="records")[0]
    params["n_samples_test2"] = n_samples_test2
    loaded_clf = CentroidClassifier(**params)
    return loaded_clf
