# Author: Jacob Statnekov
# Date created: Sometime in early 2016
# Python Version: 3.5

import numpy as np

class Layer:

    def __init__(self, node_count, batch_size, activation):
        self.previous_layer = None
        self.next_layer = None

        self.eta = 0.01
        self.activation = activation
        self.node_count = node_count
        self.matrix_dimensions = (batch_size, node_count)

        self.Outputs = None
        self.Weights = None
        self.Inputs = None
        self.Deltas = None

    #hidden methods --------------------

    #this method updates the weights of this layer.
    #this method is recursive and should only be called on the first layer of a network.
    def __update_weights(self, eta):
        # todo: can the weights be updated as we back propagate?
        if not self.is_output_layer():
            W_grad = -1*self.eta*(self.next_layer.get_deltas().dot(self.Outputs)).T
            self.Weights += W_grad
            self.next_layer.__update_weights(eta)

    #public methods --------------------


    #configure should be called once all of the adjoining layers are linked through
    #the next_layer and previous_layer members.
    def configure(self):

        self.Outputs = np.zeros(self.matrix_dimensions)

        if self.previous_layer != None:
            self.Inputs = np.zeros(self.matrix_dimensions)
            self.Deltas = np.zeros(self.matrix_dimensions)

        if not self.is_output_layer():
            next_layer_node_count = self.next_layer.node_count
            if not self.next_layer.is_output_layer():
                next_layer_node_count -= 1 #remove the bias node

            self.Weights = np.random.normal(size=[self.node_count, next_layer_node_count], scale=1E-2)

    def is_output_layer(self):
        return self.next_layer == None

    #this should be called once appropriate errors have been calculated for yhat.
    #this should be called after forward propagation when training a network.
    #this method is recursive and should only be called on the last layer of a network.
    def backpropagate(self, yhat, labels):
        if self.is_output_layer():
            self.Deltas = (yhat - labels).T

        #we can't calculate the deltas on the output layer and we don't need to
        #calculate the deltas on the input layer
        if (not self.is_output_layer()) and self.previous_layer != None :
            W_nobias = self.Weights[0:-1, :]
            self.Deltas = W_nobias.dot(self.next_layer.get_deltas()) * self.activation.DerivativeCalculate(self.Inputs).T

        if self.previous_layer != None :
            self.previous_layer.backpropagate(yhat, labels)

        if self.previous_layer == None:
            self.__update_weights(self.eta)

    #this takes a set of inputs and tests the current network with them.
    #this method is recursive and should only be called on the first layer of a network.
    def forward_propagate(self, data):
        if self.is_output_layer():
            e_inputs = np.exp(self.Inputs)
            e_inputsSum = np.sum(e_inputs, axis=1)
            Z = e_inputsSum.reshape(e_inputsSum.shape[0], 1)
            self.Outputs = e_inputs / Z
            return self.Outputs

        output_data = None

        if self.previous_layer == None:
            output_data = data
        else:
            output_data = self.activation.Calculate(self.Inputs)

        biasColumn = np.ones((output_data.shape[0], 1))
        self.Outputs = np.append(output_data, biasColumn, axis=1)

        self.next_layer.Inputs = self.Outputs.dot(self.Weights)
        return self.next_layer.forward_propagate(data)

    #this method allows for variation in the way that deltas are calculated and created
    #in subclasses
    def get_deltas(self):
        return self.Deltas




