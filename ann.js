var fs = require('fs')

const sigmoid = x => {
	return 1 / (1 + Math.E ** -x)
}

function shuffle(a) {
	for (let i = a.length - 1; i > 0; i--) {
		const j = Math.floor(Math.random() * (i + 1))
		;[a[i], a[j]] = [a[j], a[i]]
	}
	return a
}

const backprop = require('./backprop')

/**
 * Calculates the activations of the next layer in forward propogation (recursive)
 * @param {[Matrix]} weights NN weight matrices
 * @param {[Vector]} biases NN bias vectors
 * @param {Vector} activations Input activations
 * @param {Number} currentLayer Forward propogation layer progress
 */
const runANN = (weights, biases, activations, currentLayer = 0) => {
	// if this is the output layer, return the activations
	if (currentLayer >= weights.length) {
		return [activations]
	}

	// get the weights and biases relevant to the connections between this layer and the next
	const weightMatrix = weights[currentLayer]
	const biasVector = biases[currentLayer]

	// calculate the next layer's activations
	var newActivations = []
	for (var r = 0; r < weightMatrix[0].length; r++) {
		var sum = 0
		for (var c = 0; c < weightMatrix.length; c++) {
			sum += weightMatrix[c][r] * activations[c]
		}
		newActivations.push(sigmoid(sum + biasVector[r]))
	}

	// recursively propogate forward
	const feedForward = runANN(weights, biases, newActivations, currentLayer + 1)

	// combine the result activations
	const cumulativeActivations = [activations]
	feedForward.forEach(vector => {
		cumulativeActivations.splice(cumulativeActivations.length, 0, vector)
	})

	return cumulativeActivations
}

/**
 * Returns the index with the highest activation
 * @param {Vector} activations Activations of the output layer
 */
const maxActivation = activations => {
	var maxIndex = 0
	var max = Number.MIN_SAFE_INTEGER

	activations.forEach((activation, index) => {
		if (activation > max) {
			max = activation
			maxIndex = index
		}
	})

	return maxIndex
}

/**
 * Calculates the cost of a forward propogation of a neural network
 * @param {Vector} v1 Neural network output activations
 * @param {Vector} v2 Expected activations
 */
const calcCost = (v1, v2) => {
	var sum = 0

	for (var i = 0; i < v1.length; i++) {
		sum += (v1[i] - v2[i]) ** 2
	}

	return Math.sqrt(sum / (v1.length - 1))
}

/**
 *
 * @param {[Matrix]} weights NN weights
 * @param {[Vector]} biases NN biases
 * @param {[Test]} trainingSet Tests used for training
 */
const trainANN = (weights, biases, trainingSet, learnRate) => {
	trainingSet.forEach((ex, index) => {
		const actual = Object.keys(ex)[0]
		const output = runANN(weights, biases, ex[actual])

		// create the expected output
		var expected = []
		for (var i = 0; i < biases[biases.length - 1].length; i++) {
			expected.push(0)
		}
		expected[actual] = 1

		// backprop to nudge weights and biases
		backprop(output, expected, weights, biases, biases.length, learnRate)
	})
}

/**
 * Tests the neural network and generates an average cost
 * @param {[Matrix]} weights Array of weight matrices
 * @param {[Vector]} biases Array of bias vectors
 * @param {[Test]} testingSet Array of tests
 */
const testANN = (weights, biases, testingSet) => {
	console.log('Testing...')
	const numOutputs = biases[biases.length - 1].length
	var totalCost = 0
	var correct = 0
	testingSet.forEach((ex, index) => {
		const actual = Object.keys(ex)[0]

		// get the activations after forward propagation
		const output = runANN(weights, biases, ex[actual])

		// get the result
		const result = maxActivation(output[output.length - 1])
		if (actual == result) correct++

		// create the expected output activations
		const actualActivations = []
		for (var i = 0; i < numOutputs; i++) actualActivations[i] = 0
		actualActivations[actual] = 1

		// calculate cost
		const cost = calcCost(actualActivations, output[output.length - 1])

		totalCost += cost
	})

	// average the cost
	return {
		avgCost: totalCost / testingSet.length,
		correct: correct
	}
}

const spread = 1.125
/**
 * Generates an ANN with the given layer structure with random weights and biases
 * @param {[Number]} layers Numbers of neurons in each layer
 */
const ANN = layers => {
	if (!layers) layers = []
	var weights = []
	var biases = []

	// generate weights and biases
	for (var i = 0; i < layers.length - 1; i++) {
		var weightMatrix = []
		var biasVector = []

		// there should be layers[i + 1] rows and layers[i] columns
		for (var c = 0; c < layers[i]; c++) {
			var column = []
			for (var r = 0; r < layers[i + 1]; r++) {
				column.push(Math.random() * spread - spread / 2)
			}
			weightMatrix.push(column)
		}

		for (var j = 0; j < layers[i + 1]; j++) {
			biasVector.push(Math.random() * spread - spread / 2)
		}

		weights.push(weightMatrix)
		biases.push(biasVector)
	}

	return {
		weights: weights,
		biases: biases,
		calculate: function(activations) {
			return runANN(this.weights, this.biases, activations)
		},
		train: function(trainingSet, epochs, testingSet) {
			var learnRate = 0.08
			for (var i = 0; i < epochs; i++) {
				var now = Date.now()
				console.log('Epoch ' + (i + 1) + '...')
				shuffle(trainingSet)
				trainANN(this.weights, this.biases, trainingSet, learnRate)
				console.log(
					'Epoch ' + (i + 1) + ' complete: ' + (Date.now() - now) / 1000 + 's'
				)

				shuffle(testingSet)
				const testResult = testANN(this.weights, this.biases, testingSet)
				console.log(testResult)

				if (testResult.correct > 8800)
					this.save(testResult.correct + '-' + testResult.avgCost)

				learnRate = testResult.avgCost / 3
			}
		},
		test: function(testingSet) {
			return testANN(this.weights, this.biases, testingSet)
		},
		save: function(name) {
			fs.writeFile(
				__dirname + '/models/' + name + '.json',
				JSON.stringify(this),
				function(err) {
					if (err) {
						return console.log(err)
					}

					console.log('Model Saved')
				}
			)
		},
		open: function(filename) {
			const raw = fs.readFileSync(filename, 'utf8')

			const obj = JSON.parse(raw)

			this.weights = obj.weights
			this.biases = obj.biases
		}
	}
}

module.exports = ANN
