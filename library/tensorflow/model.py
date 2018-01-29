"""
Implementation of basic neural network using tensorflow framework.
"""
import os
import numpy as np

import tensorflow as tf


class Model():
    """
    Tensorflow ann model.
    """
    def __init__(self):
        """
        Init the model instance.
        """

    def predict(self, features):
        """
        Predict.

        Args:
            features (ndarray)
        """

    def fit(self,
            features,
            truth,
            input_dim,
            output_dim,
            batch_size,
            epochs):
        """
        Fit.

        Args:
            features (ndarray)
            truth (ndarray)
            input_dim (int)
            output_dim (int)
            batch_size (int)
            epochs (int)
        """
        # Network Parameters
        n_hidden_1 = 10 # 1st layer number of neurons
        n_hidden_2 = 15 # 2nd layer number of neurons

        # tf Graph input
        x_graph_value = tf.placeholder("float", [None, input_dim])
        y_graph_value = tf.placeholder("float", [None, output_dim])

        # Store layers weight & bias
        weights = {
            'h1': tf.Variable(np.ones((input_dim, n_hidden_1)).astype(np.float32)),
            'h2': tf.Variable(np.ones((n_hidden_1, n_hidden_2)).astype(np.float32)),
            'out': tf.Variable(np.ones((n_hidden_2, output_dim)).astype(np.float32))
        }
        biases = {
            'b1': tf.Variable(np.ones(n_hidden_1).astype(np.float32)),
            'b2': tf.Variable(np.ones(n_hidden_2).astype(np.float32)),
            'out': tf.Variable(np.ones(output_dim).astype(np.float32))
        }

        # Construct model
        prediction = self.neural_net(x_graph_value, weights, biases)

        # Define cost
        cost = tf.reduce_mean(tf.abs(tf.subtract(prediction, y_graph_value)))

        # Define optimizer
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.005)

        # Define train
        train = optimizer.minimize(cost)

        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()

        # Training
        with tf.Session() as sess:
            # Init
            sess.run(init)

            # Loop for each epoch
            for step in list(range(epochs)):

                # Do the batches
                batches_x, batches_y, int_division = self.create_batches(
                    features, truth, batch_size)

                # Reshape
                batches_x = np.array(batches_x).reshape((int_division, batch_size, input_dim))
                batches_y = np.array(batches_y).reshape((int_division, batch_size, output_dim))

                # Set empty list
                loss = []
                # For each batch of the list of batches
                for (batch_x, batch_y) in zip(batches_x, batches_y):
                    # Optimize
                    sess.run(train, feed_dict={x_graph_value: batch_x, y_graph_value: batch_y})

                    # Compute the cost at this stage
                    loss.append(sess.run(cost,
                                         feed_dict={x_graph_value: batch_x, y_graph_value: batch_y}))

                # Display results
                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{}".format(np.mean(loss)))

            # Over
            print("Optimization Finished!")

    @staticmethod
    def create_batches(inputs_epoch, truth_epochs, batch_size):
        """Create batches."""
        inputs_batch_list = []
        truth_batch_list = []
        epoch_len = inputs_epoch.shape[0]
        int_division = int(epoch_len/batch_size)
        # For each batch
        for b in range(int_division):
            inputs_batch_list.append(inputs_epoch[b*batch_size:(b+1)*batch_size])
            truth_batch_list.append(truth_epochs[b*batch_size:(b+1)*batch_size])
        # # Last batch
        # inputs_batch_list.append(inputs_epoch[int_division*batch_size:])
        # truth_batch_list.append(truth_epochs[int_division*batch_size:])
        return inputs_batch_list, truth_batch_list, int_division


    @staticmethod
    def neural_net(x_value, weights, biases):
        """
        Neural network.
        """
        # Hidden fully connected layer with 256 neurons
        layer_1 = tf.add(tf.matmul(x_value,
                                   weights['h1']),
                         biases['b1'])

        # Activation function
        layer_1_bis = tf.sigmoid(layer_1,
                                 name=None)

        # Hidden fully connected layer with 256 neurons
        layer_2 = tf.add(tf.matmul(layer_1_bis,
                                   weights['h2']),
                         biases['b2'])

         # Activation function
        layer_2_bis = tf.sigmoid(layer_2,
                                 name=None)

        # Output fully connected layer with a neuron for each class
        out_layer = tf.matmul(layer_2_bis, weights['out']) + biases['out']

        # Retunr value.
        return out_layer

    def persist_parameters(self, model_version):
        """
        Persist parameters.
        """
        # Set folder of the param
        folder_path = "library/keras/params/{}/".format(model_version)
        # Create
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)
        # Save

    def load_parameters(self, model_version):
        """
        Load parameters model.
        """
       # Set folder of the param
        folder_path = "library/keras/params/{}/".format(model_version)
        # Load the params
