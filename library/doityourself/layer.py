"""This module implements the structure of the layer
"""

import numpy as np


class Layer():

    def __init__(self, units, name, activation_function):

        self.units = units
        self.output = np.zeros([1, self.units])
        self.name = name
        self.create_layer_biais()
        self.initialize_layer_biais()
        self.activation = activation_function

    def create_layer_weights(self, input_dim, monitor=False):

        self.input_dim = input_dim
        self.weight_matrix = np.zeros([self.input_dim, self.units])

        if monitor:
            print('Weight matrix shape of {} : \
                {}'.format(self.name, self.weight_matrix.shape))
            print('Grad weight matrix shape of {} : \
                {}'.format(self.name, self.grad_weight_matrix.shape))

    def create_grad_batch(self, batch_len):
        self.grad_weight_batch = np.zeros([batch_len,
                                           self.weight_matrix.shape[0],
                                           self.weight_matrix.shape[1]])
        self.grad_biais_batch = np.zeros([batch_len,
                                          self.biais_matrix.shape[0],
                                          self.biais_matrix.shape[1]])

    def initialize_layer_weights(self):
        self.weight_matrix = np.ones([self.weight_matrix.shape[0],
                                      self.weight_matrix.shape[1]])

    def create_layer_biais(self, monitor=False):

        self.biais_matrix = np.zeros([1, self.units])

        if monitor:
            print('Biais matrix shape of {} : \
                {}'.format(self.name, self.biais_matrix.shape))

    def initialize_layer_biais(self):
        self.biais_matrix = np.ones([self.biais_matrix.shape[0],
                                     self.biais_matrix.shape[1]])

    @staticmethod
    def dot_biais_function(x, weight_matrix, biais_matrix):
        y = np.dot(x, weight_matrix) + biais_matrix
        return y

    def activation_function(self, x, monitor=False):
        if self.activation == 'linear':
            y = x
        elif self.activation == 'sigmoid':
            y = 1/(1+np.exp(-1*x))
        elif self.activation == 'tanh':
            y = (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

        if monitor:
            print('x activation : \n{}'.format(x))
            print('y activation : \n{}'.format(y))

        return y

    def d_x_activation_function(self, x):

        if self.activation == 'linear':
            y = np.ones([x.shape[0], x.shape[1]])
        elif self.activation == 'sigmoid':
            y = np.exp(-1*x)/np.multiply(1+np.exp(-1*x), 1+np.exp(-1*x))
        elif self.activation == 'tanh':
            m = (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
            y = 1 - np.multiply(m, m)
        return y

    def layer_forward(self, x, monitor=False):

        self.input = x.copy()
        array_dot_biais = self.dot_biais_function(
            self.input,
            self.weight_matrix,
            self.biais_matrix)
        self.output = self.activation_function(array_dot_biais)

        if monitor:
            print('Input of {} : \n{}'.format(self.name, self.input))
            print('Computed output of {} : \n{}'.format(
                self.name,
                self.output))

        return self.output

    def evaluate_layer_grad(self, g_upstream):

        self.g_act = self.evaluate_layer_act_grad()
        self.grad_weight_matrix = self.evaluate_layer_weight_grad(g_upstream)
        self.grad_biais_matrix = self.evaluate_layer_biais_grad(g_upstream)

    def evaluate_layer_act_grad(self):

        g_act = self.d_x_activation_function(self.dot_biais_function(
            self.input,
            self.weight_matrix,
            self.biais_matrix))
        return g_act

    def evaluate_layer_weight_grad(self, g_upstream, monitor=False):

        grad_weight_matrix = np.zeros([self.input_dim, self.units])
        for i in range(self.input_dim):
            for j in range(self.units):
                grad_weight_matrix[i, j] = \
                    g_upstream[0, j]*self.g_act[0, j]*self.input[0, i]
        if monitor:
            print('Gradient weight matrix : {}'.format(grad_weight_matrix))
            print('Gradient weight matrix shape : \
                {}'.format(grad_weight_matrix.shape))
        return grad_weight_matrix

    def evaluate_layer_biais_grad(self, g_upstream, monitor=False):

        grad_biais_matrix = np.zeros([1, self.units])
        for j in range(self.units):
            grad_biais_matrix[0, j] = g_upstream[0, j]*self.g_act[0, j]

        if monitor:
            print('Gradient biais matrix : {}'.format(grad_biais_matrix))
            print('Gradient biais matrix shape : \
                {}'.format(grad_biais_matrix.shape))

        return grad_biais_matrix

    def evaluate_layer_downstream_grad(self, g_upstream, monitor=False):
        '''
        g_downstream is a vector of dimension [input_dim, 1]
        '''
        g_downstream = np.zeros([1, self.input_dim])
        for i in range(self.input_dim):
            for j in range(self.units):
                g_downstream[0, i] += \
                    self.weight_matrix[i, j]*g_upstream[0, j]*self.g_act[0, j]

        if monitor:
            print('g_downstream of {} : {}'.format(self.name, g_downstream))

        return g_downstream

    def update_layer_params(self, learning_rate):

        self.weight_matrix = self.weight_matrix - \
            learning_rate*self.grad_weight_matrix
        self.biais_matrix = self.biais_matrix - \
            learning_rate*self.grad_biais_matrix
