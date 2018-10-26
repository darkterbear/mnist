const FNN = require('./fnn')
const MNIST = require('./mnist/mnist')

const nn = FNN.create([784, 15, 10])

nn.train(MNIST.trainingSet, 20, 50, 1, MNIST.testSet)
