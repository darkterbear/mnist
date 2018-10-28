const ANN = require('./ann')
const MNIST = require('./mnist/mnist')

var nn = ANN()
nn.open(__dirname + '/models/9373-0.1041.json')

// var nn = ANN([784, 16, 16, 10])

// nn.train(MNIST.trainingSet, 20, MNIST.testSet)

const testResults = nn.test(MNIST.testSet)

console.log(testResults)
