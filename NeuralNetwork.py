# Author: Jacob Statnekov
# Date created: Sometime in early 2016
# Python Version: 3.5


import numpy as np
import Layer as ly
import Activation as ac

_debug_view_training_error_per_batch = False

class NeuralNetwork:

    def __init__(self, list_of_node_counts, batch_size):
        self.layers = []

        num_layers = len(list_of_node_counts)

        print("   Your specified architecture : ")
        for i in range(num_layers):

            activation = ac.SigmoidActivationFunction
            node_count = list_of_node_counts[i]

            if i != num_layers -1:
                node_count = node_count+1 #add the bias

            self.layers.append(ly.Layer(node_count, batch_size, activation))
            print ("    layer {} has {} nodes.".format(i, list_of_node_counts[i]))

        # do wiring
        for i in range(num_layers):
            if i != num_layers -1:
                self.layers[i].next_layer = self.layers[i+1]
            if i != 0:
                self.layers[i].previous_layer = self.layers[i-1]

        for i in range(num_layers):
            self.layers[i].configure()

    #this does forward propagation on the network and prints the error.
    #data must be a 3 level deep array. The outermost array holds the batches, the middle array
    #holds the sets of data, and the innermost array holds input values for each input layer node.
    #labels must be set up in the same way with 3 levels of array, the outermost being batches,
    #the middle being labels matching each input data item, and the innermost array marks which
    #output node is the correct node with a 1 (all other output nodes are 0 per innermost array).
    def evaluate_all_data(self, data, labels):
        accumulated_error = 0

        for i in range(len(data)):
            #this will be a list of lists of output node values
            output = self.layers[0].forward_propagate(data[i])
            #the largest value for each set of output node values is the predicted value
            yhat = np.argmax(output, axis=1)
            #the labels are a list of lists of correct output node values 
            #ex labels[0][0] = [0,0,0,1,0,0] which would mean that the class associated with the 3rd node is the correct class
            for index in range(len(yhat)):
                correct_node_index = yhat[index]
                if labels[i][index][correct_node_index] == 0:
                    accumulated_error += 1

        N = len(labels)*len(labels[0])
        return (float(accumulated_error)/N)

    #this does forward propagation on the network and prints the error.
    #data must be a 2 level deep array. The outermost array holds the sets of data, and the
    #innermost array holds input values for each input layer node.
    #labels must be set up in the same way with 2 levels of array, the outermost being the set
    #of possible labels and the innermost array marking which is the true label with a 1 
    #(all other output nodes are 0 per innermost array).
    def evalute_single_batch(self, data, labels):
        accumulated_error = 0

        #this will be a list of lists of output node values
        output = self.layers[0].forward_propagate(data)
        #the largest value for each set of output node values is the predicted value
        yhat = np.argmax(output, axis=1)
        #the labels are a list of lists of correct output node values 
        #ex labels[0][0] = [0,0,0,1,0,0] which would mean that the class associated with the 3rd node is the correct class
        for index in range(len(yhat)):
            correct_node_index = yhat[index]
            if labels[index][correct_node_index] == 0:
                accumulated_error += 1

        N = len(labels)
        return ("Current Error: {}".format(float(accumulated_error)/N))


    def train(self, train_data, train_labels, num_epochs=5):
        self.train_and_validate(train_data, train_labels, test_data=None, test_labels=None, num_epochs=5)

    #train_data is 3 layers of nested array; the first layer is the batches, the layer within
    #that is the array of different data, and the array within that is the actual data that
    #matches each input node. train_labels is 3 levels of array, just like the train_data; the
    #outermost is the array of batches, the array within that holds individual label arrays,
    #and each label array holds output node data that is either 0 or 1. The current 
    #classification scheme only allows one output classification, so only one node can be
    #marked as 1 within each of the innermost arrays.
    def train_and_validate(self, train_data, train_labels, test_data, test_labels, num_epochs=5):

        passes_validation = (len(train_data) == len(train_labels)) 

        if passes_validation:
            for data_i, label_i in zip(train_data, train_labels):
                if len(data_i) != len(label_i):
                    passes_validation = false
                    break

        if passes_validation and test_data is not None and test_labels is not None:
            passes_validation = passes_validation and (len(test_data) == len(test_labels))

        if passes_validation and test_data is not None and test_labels is not None:
            for data_i, label_i in zip(test_data, test_labels):
                if len(data_i) != len(label_i):
                    passes_validation = false
                    break

        if not passes_validation:
            print ("The data and labels do not have matching sizes. There must be a label for each input to the network.")
            return

        pre_train_error = self.evaluate_all_data(train_data, train_labels)
        print ("Pre-Training Train Data Error : {}".format(pre_train_error))

        if test_data is not None:
            pre_train_validation_error = self.evaluate_all_data(test_data, test_labels)
            print ("Pre-Training Validation Data Error : {}".format(pre_train_validation_error))

        print ("Training {} batches for {} epochs".format(len(train_data), num_epochs))
        for t in range(num_epochs):
            print ("[{0:4d}]---------------------".format(t))

            for i in range(len(train_data)):
                output = self.layers[0].forward_propagate(train_data[i])
                self.layers[-1].backpropagate(output, train_labels[i])

                if( _debug_view_training_error_per_batch ):
                    print(self.evalute_single_batch(train_data[i], train_labels[i]))

            train_error = self.evaluate_all_data(train_data, train_labels)
            print ("Train Data Error : {}".format(train_error))

            if test_data is not None:
                validation_error = self.evaluate_all_data(test_data, test_labels)
                print ("Validation Data Error : {}".format(validation_error))

    #this saves the serialized weights to each layer in a file
    def save(self, path):
        with open(path, "w+") as f:
            pickle.dump(map(lambda x: x.Weights, self.layers), f)

    #this loads the serialized weights from a file
    def load(self, path):
        with open(path, "r") as f:
            data = pickle.load(f)

        valid = True
        if len(data) != len(self.layers):
            valid = False
        else:
            for i in range(len(data)):
                if data[i] != None and len(data[i]) != len(self.layers[i].Weights):
                    valid = False
                    break

        if not valid:
            raise Exception("Invalid layer dimensions")

        for i in range(len(self.layers)):
            self.layers[i].Weights = data[i]


