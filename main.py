# Author: Jacob Statnekov
# Date created: Sometime in early 2016
# Python Version: 3.5

import DataPreparation as DP
import NeuralNetwork as NN
import pickle, gzip, sys



if(len(sys.argv) != 2):
    print("Please provide the locatation of your mnist.pkl.gz file as an argument to this script.")
    exit()

mnist_path = sys.argv[1]
print("Begining Data Loading from {} ...".format(mnist_path))
f = gzip.open(mnist_path, 'rb')


pickle_loader = pickle._Unpickler(f)
pickle_loader.encoding = 'latin1'
train_set, valid_set, test_set = pickle_loader.load()
f.close()


train_set_data = train_set[0]
train_set_labels = train_set[1]
valid_set_data = valid_set[0]
valid_set_labels = valid_set[1]
test_set_data = test_set[0] #todo do something with this
test_set_labels = test_set[1] #todo do something with this


minibatch_size = 26 #any size will do
train_data, train_labels = DP.DataPreparation.format_minibatch_data(train_set_data, train_set_labels, minibatch_size)
valid_data, valid_labels = DP.DataPreparation.format_minibatch_data(valid_set_data, valid_set_labels, minibatch_size)


print("Data Loading Complete")
print("Begining Training...")

archetecture = [784, 100, 100, 10]
nn = NN.NeuralNetwork(archetecture, minibatch_size)
nn.train_and_validate( train_data, train_labels, valid_data, valid_labels)

print("Training Complete")



exit()
