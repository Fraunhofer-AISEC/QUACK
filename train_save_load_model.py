"""
Copyright 2024 Fraunhofer AISEC: Kilian Tscharke
"""

from centroid_classifier import CentroidClassifier, load_model

# set parameters and train model
n_qubits = 3
n_layers = 53
lr_ka = 0.5
lr_co = 1.
decay_rate = 0.9
reg_param_ka = 0.001
reg_param_co = 0.001
init_weights_scale = 0.1
dataset = "census_income"

epochs = 2
epochs_ka = 1
epochs_co = 1
n_samples_train = 100
n_samples_test = 40
gpu=False


clf = CentroidClassifier(n_layers, n_qubits, init_weights_scale,
                         dataset, n_samples_train, n_samples_test,
                         reg_param_ka, reg_param_co, epochs, epochs_ka, epochs_co, lr_ka, lr_co, decay_rate, gpu,
                         silent=False)
clf.train()

# save model
model_id = clf.save_model()
print(f"ID of saved model: {model_id}")


# load model and set number of test2 samples for evaluation
loaded_clf = load_model(model_id, n_samples_test2=400)


# evaluate the model on the test2 set
loaded_clf.eval_model()
# save the results on the test2 set to the models_test2.csv file
loaded_clf.save_model(model_id)
