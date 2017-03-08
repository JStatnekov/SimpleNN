# Author: Jacob Statnekov
# Date created: Sometime in early 2016
# Python Version: 3.5

#This class is not exactly pythonic since it uses static methods instead of top level functions
class DataPreparation:

    @staticmethod
    def format_batch_data(data, labels):
        batch_size = 1
        return DataPreparation.create_minibatches(data, labels, batch_size)

    @staticmethod
    def format_online_data(data, labels):
        batch_size = len(data)
        return DataPreparation.create_minibatches(data, labels, batch_size)

    @staticmethod
    def format_minibatch_data(data, labels, batch_size):

        import numpy as np #sneaky import!

        number_of_classes = len(set(labels))
        N = data.shape[0]
        remainder = N % batch_size

        if remainder != 0:
            print ("{} does not evenly divide {}! The last batch will only have {} data points".format( batch_size, N, remainder))
        if N < batch_size:
            print("There are not enough entries in your dataset to satisfy a single batch of the size specified. Perhaps something is wrong?")
            return [],[]

        chunked_data = []
        chunked_labels = []
        idx = 0
        while idx + batch_size <= N + remainder:

            endpoint_of_slice = idx + batch_size
            if endpoint_of_slice > N:
                endpoint_of_slice = N

            batch_labels = labels[idx:endpoint_of_slice]

            #This will mostly be the size of batch_size.
            #In the case where a batch size does not divide the data
            #evently then we want to still capture the final data points
            #and this will be the size of the remainder
            number_of_batch_labels_in_current_chunk = batch_labels.shape[0]

            #a 2D array, the first dimension is the index to the label
            #the second dimension represents the output nodes. There will
            #be only one output class and it will be marked as 1 following this step
            bv = np.zeros((number_of_batch_labels_in_current_chunk, number_of_classes))

            #the correct output node will be marked in the  as 1
            for i in range(number_of_batch_labels_in_current_chunk):
                bv[i, batch_labels[i]] = 1.0

            chunked_labels.append(bv)

            chunked_data.append(data[idx:endpoint_of_slice, :])

            idx += batch_size

        return chunked_data, chunked_labels


