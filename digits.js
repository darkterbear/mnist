const ANN = require('./ann')
const MNIST = require('./mnist/mnist')
const parsePNG = require('./pngparser')

// IMPORT
var nn = ANN()

// nn.open(__dirname + '/models/9071-0.1501.json')

// TRAIN
var nn = ANN([784, 16, 16, 10])

nn.train(MNIST.trainingSet, 1000, MNIST.testSet)

// TEST
const testResults = nn.test(MNIST.testSet)

console.log(testResults)

// CALCULATE SINGLE
// nn.run(MNIST.testSet[0])

// CALCULATE FROM FILE
// parsePNG(__dirname + '/images/6.png', pixels => {
// 	//console.log(JSON.stringify(MNIST.testSet[0]))

// 	const obj = {}
// 	obj['6'] = pixels

// 	//console.log(JSON.stringify(obj))
// 	nn.run(obj)
// })
