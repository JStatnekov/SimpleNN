This is a simple neural network that I wrote that includes the logic for forward and backward propagation in the layer classes. At the time I wrote this I had seen a few implementations that used a Mediator to manage the training and wanted to try a different design that minimized the number of supporting classes involved in training. 


As I saw it, the advantage to having all of the logic for forward and backward propagation live within each layer was that layers with dramatically different training needs could be interfaced with one another without needing to write lots of specific cases into one (or many) training Mediators. In practice, the number of different layer types isn't all that large so Mediators are actually a good idea (unless the point is to experiment with lots of weird layers). This is especially true when complex interactions from things like residual layers start making their way into the code. I'm pretty sure coding a skip layer for the design pattern I've used here would be more complex than the complexity of having a Mediator 'remember' to add in the output from a few layers previous. There's a lot of middle ground between placing the training and execution logic in the layers and placing it in a helper, this is just one side of the spectrum.



To run this script just supply the location of your pickled MNIST data as a parameter and let it run (data can be downloaded from http://deeplearning.net/data/mnist/mnist.pkl.gz). Most of the things that you'd want to configure are managable from the main file. Here's a typical output from running the program:


Begining Data Loading from E:/MNIST/mnist.pkl.gz ...
26 does not evenly divide 50000! The last batch will only have 2 data points
26 does not evenly divide 10000! The last batch will only have 16 data points
Data Loading Complete
Begining Training...
   Your specified architecture :
    layer 0 has 784 nodes.
    layer 1 has 100 nodes.
    layer 2 has 100 nodes.
    layer 3 has 10 nodes.
Pre-Training Train Data Error : 0.9009760390415616
Pre-Training Validation Data Error : 0.9023976023976024
Training 1923 batches for 5 epochs
[   0]---------------------
Train Data Error : 0.6614264570582823
Validation Data Error : 0.6592407592407592
[   1]---------------------
Train Data Error : 0.1797071882875315
Validation Data Error : 0.16123876123876124
[   2]---------------------
Train Data Error : 0.0967238689547582
Validation Data Error : 0.08561438561438561
[   3]---------------------
Train Data Error : 0.07302292091683667
Validation Data Error : 0.06723276723276723
[   4]---------------------
Train Data Error : 0.058062322492899714
Validation Data Error : 0.05284715284715285
Training Complete