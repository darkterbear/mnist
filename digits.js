const FNN = require('./fnn')
const MNIST = require('./mnist/mnist')

const nn = FNN.create([784, 16, 16, 10])

// console.log(nn.feedForward(MNIST.testSet[0][Object.keys(MNIST.testSet[0])[0]]))
// console.log(nn.calculate(MNIST.testSet[0][Object.keys(MNIST.testSet[0])[0]]))

nn.train(MNIST.trainingSet, 1, 50, 1, MNIST.testSet)
// console.log(nn)
