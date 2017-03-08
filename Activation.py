# Author: Jacob Statnekov
# Date created: Sometime in early 2016
# Python Version: 3.5

import numpy as np

class ActivationFunction :
    @staticmethod
    def Calculate(x):
        print ("not implemented")
    @staticmethod
    def DerivativeCalculate(x):
        print ("not implemented")

class SigmoidActivationFunction(ActivationFunction):
    @staticmethod
    def Calculate(x):
        return 1 / (1 + np.exp(-x))
    @staticmethod
    def DerivativeCalculate(x):
        temp = SigmoidActivationFunction.Calculate(x)
        return temp*(1 - temp)