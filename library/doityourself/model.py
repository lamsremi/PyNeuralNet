"""
Do it yourself Naive Bayes model.

TODO:
    * Store output dim for use in prediction.
"""
import os
import pickle
import math

import numpy as np

from library.doityourself.layer import Layer


class Model():
    """
    Do it yourself class model.
    """
    def __init__(self):
        """Init the model."""
        self._layers_list = []
        self._input_dim = None
        self._output_dim = None # [To be stored]

    def predict(self, inputs_epoch):
        """Predict method.

        Args:
            input_var (Serie or dict): input to predict the class from.
        """
        # print(inputs_epoch.shape)
        epoch_len = inputs_epoch.shape[0]
        prediction = np.zeros([epoch_len, self._output_dim])
        for i in range(epoch_len):
            inputs_epoch_i = np.reshape(inputs_epoch[i, :],
                                        [1, inputs_epoch.shape[1]])
            prediction[i, :] = self.propagate_forward(inputs_epoch_i)
        return prediction

    def propagate_forward(self, input):
        """
        Propagate forward.
        """
        input_layer = input.copy()
        for layer in self._layers_list:
            output_layer = layer.layer_forward(input_layer)
            input_layer = output_layer.copy()
        output = output_layer.copy()
        return output

    def fit(self, inputs_epoch, truth_epoch, input_dim, output_dim, batch_size, epochs):
        """Fit."""
        # Set input and output dim.
        self._input_dim = input_dim
        self._output_dim = output_dim

        self.add_layer(Layer(10, 'Layer 1', "sigmoid"))
        self.add_layer(Layer(15, 'Layer 2', "sigmoid"))
        self.add_layer(Layer(self._output_dim, 'Layer output', 'linear'))
        self.compile(loss="mean_absolute_error", learning_rate=0.005)
        history_cost = np.zeros([epochs, 1])
        for r_value in range(epochs):
            print('\nFitting for epoch {}...'.format(r_value))
            history_cost[r_value, 0] = self.fit_epoch(
                inputs_epoch, truth_epoch, batch_size)
            print('Cost epoch {} : {}'.format(r_value, history_cost[r_value]))

    def persist_parameters(self, model_version):
        """
        Persist the model parameters..
        """
        # Set folder of the param
        folder_path = "library/doityourself/params/{}/".format(model_version)
        # Create
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)
        # Save layer
        with open(folder_path + "/layers.pkl", 'wb') as handle:
            pickle.dump(self._layers_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # Save outputdim
        with open(folder_path + "/output_dim.pkl", 'wb') as handle:
            pickle.dump(self._output_dim, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_parameters(self, model_version):
        """
        Load parameters model.
        """
       # Set folder of the param
        folder_path = "library/doityourself/params/{}/".format(model_version)
        # Load the params
        with open(folder_path + "/layers.pkl", 'rb') as handle:
            self._layers_list = pickle.load(handle)
        # Load output dim
        with open(folder_path + "/output_dim.pkl", 'rb') as handle:
            self._output_dim = pickle.load(handle)

    def add_layer(self, layer_instance):
        """Add layer."""
        self._layers_list.append(layer_instance)

    def compile(self, loss, learning_rate):
        """Compile."""
        self.create_model_weights()
        self.loss = loss
        self.learning_rate = learning_rate

    def create_model_weights(self):
        """Create model weights."""
        previous_layer_output_dim = self._input_dim
        for layer in self._layers_list:
            layer.create_layer_weights(previous_layer_output_dim)
            layer.initialize_layer_weights()
            previous_layer_output_dim = layer.units

    @staticmethod
    def cost_function(x, truth, loss):
        """Cost function."""
        if loss == 'mse':
            return (x-truth)**2
        elif loss == 'mean_absolute_error':
            return np.abs(x-truth)

    @staticmethod
    def d_x_cost_function(x, truth, loss):
        """d_x cost function."""
        if loss == 'mse':
            return 2*(x - truth)
        elif loss == 'mean_absolute_error':
            return np.sign(x-truth)

    def evaluate_model_grad(self, feature, truth):
        """
        Evaluate model grad.
        """
        pred = self.propagate_forward(feature)
        g_upstream = np.array(self.d_x_cost_function(pred, truth, self.loss))
        for layer in self._layers_list[::-1]:
            layer.evaluate_layer_grad(g_upstream)
            layer.g_downstream = \
                layer.evaluate_layer_downstream_grad(g_upstream)
            g_upstream = layer.g_downstream

    def evaluate_batch_model_grad(self, inputs_batch, truth_batch):
        """Evaluate batch model grad."""
        # print('\n----> Evaluation of the gradient of the batch...')
        batch_len = inputs_batch.shape[0]
        self.create_model_grad_batch(batch_len)
        self.evaluation_sample_batch_grad(batch_len, inputs_batch, truth_batch)
        self.compute_overall_grad()

    def create_model_grad_batch(self, batch_len):
        """Create model grad batch."""
        for layer in self._layers_list:
            layer.create_grad_batch(batch_len)

    def evaluation_sample_batch_grad(self, batch_len,
                                     inputs_batch, truth_batch):
        """Evaluate sample batch grad."""
        for i in range(batch_len):
            inputs_batch_i = np.reshape(inputs_batch[i],
                                        [1, inputs_batch.shape[1]])
            truth_batch_i = np.reshape(truth_batch[i],
                                       [1, truth_batch.shape[1]])
            self.evaluate_model_grad(inputs_batch_i, truth_batch_i)
            for layer in self._layers_list:
                layer.grad_weight_batch[i, :, :] = layer.grad_weight_matrix
                layer.grad_biais_batch[i, :, :] = layer.grad_biais_matrix

    def compute_overall_grad(self):
        """Compute overall grad."""
        for layer in self._layers_list:
            layer.grad_weight_matrix = np.mean(layer.grad_weight_batch, axis=0)
            layer.grad_biais_matrix = np.mean(layer.grad_biais_batch, axis=0)

    def update_model_params(self):
        """Update model param."""
        # print('\n----> Updating of the model for the provided batch...')
        for layer in self._layers_list:
            layer.update_layer_params(self.learning_rate)

    def fit_batch(self, inputs_batch, truth_batch):
        """Fit batch."""
        self.evaluate_batch_model_grad(inputs_batch, truth_batch)
        self.update_model_params()

    def create_batches(self, inputs_epoch, truth_epochs, batch_size):
        """Create batches."""
        # print('\n--> Creation of batches...')
        inputs_batch_list = []
        truth_batch_list = []
        epoch_len = inputs_epoch.shape[0]
        int_division = int(epoch_len/batch_size)
        for b in range(int_division):
            inputs_batch_list.append(inputs_epoch[b*10:(b+1)*10])
            truth_batch_list.append(inputs_epoch[b*10:(b+1)*10])
        inputs_batch_list.append(inputs_epoch[int_division*10:])
        truth_batch_list.append(inputs_epoch[int_division*10:])
        return inputs_batch_list, truth_batch_list

    def fit_epoch(self, inputs_epoch, truth_epoch, batch_size):
        """Fit epoch.
        """
        inputs_batch_list, truth_batch_list = \
            self.create_batches(inputs_epoch, truth_epoch, batch_size)
        batch_mean_cost = np.zeros([len(inputs_batch_list), 1])
        for b in range(len(inputs_batch_list)):
            # print('\n--> Fitting for the batch {}...'.format(b))
            self.fit_batch(inputs_batch_list[b], truth_batch_list[b])
            pred_epoch = self.predict(inputs_epoch)
            batch_mean_cost[b] = \
                np.mean(self.cost_function(pred_epoch, truth_epoch,
                                           self.loss), axis=0)
        return np.mean(batch_mean_cost, axis=0)

