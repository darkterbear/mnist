const ANN = require('./ann')
const MNIST = require('./mnist/mnist')

var nn = ANN()

// nn.open(__dirname + '/models/8797-0.0774.json')

var nn = ANN([784, 16, 16, 10])

nn.train(MNIST.trainingSet, 200, MNIST.testSet)

const testResults = nn.test(MNIST.testSet)

if (testResults.correct > 8800) {
	nn.save(testResults.correct + '-' + testResults.avgCost.toFixed(4))
}

console.log(testResults)
