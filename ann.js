const fs = require('fs')
const utils = require('./utils')

const backprop = require('./backprop')

const feedForward = (w, b, i) => {
	var a = [i]

	for (var l = 0; l < b.length; l++) {
		const left = a[l]
		var right = []

		for (var r = 0; r < w[l][0].length; r++) {
			var sum = 0
			for (var c = 0; c < w[l].length; c++) {
				sum += w[l][c][r] * left[c]
			}
			right.push(utils.sigmoid(sum + b[l][r]))
		}
		a.push(right)
	}

	return a
}

const maxActivation = a => {
	return a.reduce((p, c, i) => (c > a[p] ? i : p), 0)
}

const calcCost = (v1, v2) => {
	var sum = 0

	for (var i = 0; i < v1.length; i++) {
		sum += (v1[i] - v2[i]) ** 2
	}

	return sum
}

const trainANN = (w, b, trainingSet, eta) => {
	trainingSet.forEach(ex => {
		const correct = Object.keys(ex)[0]
		const output = feedForward(w, b, ex[correct])

		// create the expected output
		var expected = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		// for (var i = 0; i < b[b.length - 1].length; i++) {
		// 	expected.push(0)
		// }
		expected[correct] = 1

		// backprop to nudge weights and biases
		backprop(output, expected, w, b, b.length, eta)
	})
}

const testANN = (weights, biases, testingSet) => {
	console.log('Testing...')
	const numOutputs = biases[biases.length - 1].length
	var totalCost = 0
	var correct = 0
	testingSet.forEach((ex, index) => {
		const actual = Object.keys(ex)[0]

		// get the activations after forward propagation
		const output = feedForward(weights, biases, ex[actual])

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
			return feedForward(this.weights, this.biases, activations)
		},
		train: function(trainingSet, epochs, testingSet) {
			var learnRate = 0.08
			for (var i = 0; i < epochs; i++) {
				var now = Date.now()
				console.log('Epoch ' + (i + 1) + '...')
				utils.shuffle(trainingSet)
				trainANN(this.weights, this.biases, trainingSet, learnRate)
				console.log(
					'Epoch ' + (i + 1) + ' complete: ' + (Date.now() - now) / 1000 + 's'
				)

				utils.shuffle(testingSet)
				const testResult = testANN(this.weights, this.biases, testingSet)
				console.log(testResult)

				if (testResult.correct > 9300)
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
