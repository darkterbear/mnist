const fs = require('fs')
const ANN = require('./ann')
const MNIST = require('./mnist/mnist')

var nn = ANN([784, 16, 16, 10])

const args = process.argv.slice()

const intialLearnRate = parseFloat(args[2])
const learnRateDecay = parseFloat(args[3])

const trainResults = nn.train(
	MNIST.trainingSet,
	MNIST.testSet,
	1,
	intialLearnRate,
	learnRateDecay,
	0.96
)

fs.writeFileSync(
	__dirname + '/logs/' + intialLearnRate + learnRateDecay + '.json',
	JSON.stringify(trainResults)
)
