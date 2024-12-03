"""
Copyright 2024 Fraunhofer AISEC: Kilian Tscharke
"""

import torch
import torch.nn as nn


class StateVecSimTorch(nn.Module):
    def __init__(self, n_qubits, n_layers, init_weights_scale=1., gpu=False, seed=42):
        super().__init__()
        torch.manual_seed(seed)
        self.n_qubits = n_qubits
        self.state = None
        self.n_layers = n_layers

        if gpu:
            # make ready for gpu
            try:
                self.device = torch.device("cuda")
            except:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")

        self.weights = nn.Parameter((init_weights_scale * torch.rand((n_layers, n_qubits, 3))).requires_grad_(True))
        self.bias = nn.Parameter((init_weights_scale * torch.rand((n_layers, n_qubits, 3))).requires_grad_(True))

    def get_angles(self, X):
        """
        multiplies X with weights and adds biases. Expands X such that for a single sample
        the features are encoded into the angles with repetition,
        e.g. for 2 qubits, 2 layers and 5 features:
        Rot(w1*x1+b1, w2*x2+b2, w3*x3+b3) Rot(w2*x2+b2, w3*x3+b3, w4*x4+b4)
        Rot(w4*x4+b4, w5*x5+b5, w1*x1+b1) Rot(w5*x5+b5, w1*x1+b1, w2*x2+b2)
        :param X: shape (n_samples, n_features)
        :return: angles of shape (n_samples, n_layers, n_qubits, 3)
        """
        n_samples = X.shape[0]
        n_features = X.shape[1]
        n_angles = self.n_qubits * self.n_layers * 3
        num_reps = n_angles // n_features + 1
        X_expanded = X.repeat(1, num_reps)
        X_expanded = X_expanded[:, :n_angles].reshape(n_samples, self.n_layers, self.n_qubits, 3).to(self.device)
        W = self.weights.unsqueeze(0).repeat(X.shape[0], 1, 1, 1)
        B = self.bias.unsqueeze(0).repeat(X.shape[0], 1, 1, 1)
        angles = X_expanded * W + B
        return angles

    def get_single_qubits_unitary(self, angles):
        """
        calculates states after applying a rot gate parametrized by [phi, theta, omega].
        Initial state is defined in init
        X shape = n_samples x n_qubits x 3 | (phi, theta, omega)"""
        n_qubits = angles.shape[1]
        n_samples = angles.shape[0]

        # https://docs.pennylane.ai/en/stable/code/api/pennylane.Rot.html
        # X =[phi, theta, omega] = RZ(omega)RY(theta)RZ(phi)
        ctheta = torch.cos(angles[:, :, 1] / 2)
        stheta = torch.sin(angles[:, :, 1] / 2)
        phi_plus_omega = ((angles[:, :, 0] + angles[:, :, 2]) / 2)
        phi_minus_omega = ((angles[:, :, 0] - angles[:, :, 2]) / 2)

        m00 = torch.exp(-1j * phi_plus_omega) * ctheta
        m01 = -torch.exp(1j * phi_minus_omega) * stheta
        m10 = torch.exp(-1j * phi_minus_omega) * stheta
        m11 = torch.exp(1j * phi_plus_omega) * ctheta
        M = torch.stack((m00, m01, m10, m11), dim=2).reshape(angles.shape[0], n_qubits, 2, 2)
        # kudos to SI
        M = self.get_system_unitary(M)
        return M

    def get_system_unitary(self, M):
        """
        :param M: tensor of single qubit unitaries of shape (n_samples, n_qubits, 2, 2)
        :return: M: tensor of system unitaries of shape (n_samples, 2**n_qubits, 2**n_qubits)
        """
        # kudos to SI
        n_samples = M.shape[0]
        n_qubits = self.n_qubits
        T = [chr(66 + n) if n < 25 else chr(72 + n) for n in range(51)]
        esum = ",".join([f"A{T[2 * n]}{T[2 * n + 1]}" for n in range(n_qubits)]) + \
               "->A" + \
               "".join([T[2 * n] for n in range(n_qubits)]) + \
               "".join([T[2 * n + 1] for n in range(n_qubits)])
        M = torch.einsum(esum, *(M[:, n, :, :] for n in range(n_qubits))).reshape(n_samples, 2 ** n_qubits,
                                                                                  2 ** n_qubits)
        return M

    def get_single_CNOT_matrix(self, control, target):
        """
        calculates states after applying two qubit gates
        gates: list of strings of the two qubit gates that will be applied
        qubit_map: list of tupels of qubit numbers (control, target) on which the gates will be applied, or string "ring", "next"
        """

        u = torch.zeros((2 ** self.n_qubits, 2 ** self.n_qubits))

        for i in range(2 ** self.n_qubits):
            bit_i = list(bin(i)[2:].zfill(self.n_qubits))
            if bit_i[control] == "1":
                bit_i[target] = str(int(bit_i[target]) ^ 1)
                bit_i_str = "".join(bit_i)
                index_1 = int(bit_i_str, 2)
                u[index_1, i] = 1
            else:
                u[i, i] = 1
        return u

    def Rot(self, X):
        """
        applies Rot gate on qubits.
        :param X: angles of shape n_samples x n_qubits x 3
        :return: final state after applying gate to all qubits
        """
        M = self.get_single_qubits_unitary(X)
        final_state = self.apply_unitary(M)
        return final_state


    def CNOT(self, qubit_map, n_samples):
        """
        applies CNOT gate on qubits.
        :param qubit_map: list of tupels of qubit numbers (control, target) on which the gates will be applied, or string "ring", "next"
        :return: final state after applying gate to all qubits
        """
        qubit_map_string = ""
        if qubit_map == "next" or qubit_map == "ring":
            qubit_map_string = qubit_map
            qubit_map = [(i, i + 1) for i in range(self.n_qubits - 1)]
        if qubit_map_string == "ring":
            qubit_map.append((self.n_qubits - 1, 0))
        M = torch.eye(2 ** self.n_qubits)
        for control, target in qubit_map:
            M = self.get_single_CNOT_matrix(control, target).repeat(n_samples, 1, 1)
            final_state = self.apply_unitary(M)

        return final_state


    def combine_layer_unitaries(self, unitaries):
        """
        :param unitaries: list of unitaries of shape (n_samples, 2**n_qubits, 2**n_qubits)
        :return: combined unitary of shape (n_samples, 2**n_qubits, 2**n_qubits)
        """
        unitary = torch.stack(unitaries).type(torch.complex64)
        n_layers = len(unitaries)
        # Perform batch matrix multiplication using einsum
        T = [chr(66 + n) if n < 25 else chr(72 + n) for n in range(51)]
        esum = ",".join([f"A{T[n]}{T[n + 1]}" for n in range(n_layers)]) + \
               "->A" + T[0] + T[n_layers]
        u_total = torch.einsum(esum, *(unitary[n, :, :, :] for n in range(n_layers))).type(torch.complex64)
        return u_total

    def apply_unitary(self, unitary):
        """
        applies unitary to self.state
        :param unitary: unitary of shape (n_samples, 2**n_qubits, 2**n_qubits)
        :return: final state after applying unitary
        """
        state = self.state.type(torch.complex128).to(self.device)
        unitary = unitary.type(torch.complex128).to(self.device)
        final_state = torch.einsum('ijk,ik->ij', unitary, state)
        self.state = final_state
        return final_state

    def forward(self, X):
        """
        creates circuit and applies it to self.state
        :param X: angles of shape n_samples x n_qubits x 3
        :return: final state after applying circuit
        """
        self.state = torch.zeros(2 ** self.n_qubits)
        self.state[0] = 1.
        self.state = self.state.unsqueeze(1).repeat(1, X.shape[0]).mT.type(torch.complex64)
        angles = self.get_angles(X)
        for l in range(self.n_layers):
            self.Rot(angles[:, l, :, :])
            self.CNOT("ring", n_samples=X.shape[0])

        return self.state