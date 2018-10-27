const fs = require('fs')
const utils = require('./utils')
const dS = utils.dSigmoid
const dC = utils.dCost

const backprop = (a, e, weights, biases, l, eta) => {
	var w = weights[l]
	var b = biases[l]

	var rA = a[l + 1]
	var lA = a[l]

	w.forEach((wC, k) => {
		wC.forEach((_, j) => {
			const g = lA[k] * dS(rA[j]) * dC(rA[j], e[j])

			w[k][j] -= g * eta
		})
	})

	b.forEach((_, j) => {
		const g = dS(rA[j]) * dC(rA[j], e[j])
		b[j] -= g * eta
	})

	var nextA = lA.slice()
	lA.forEach((_, k) => {
		var g = 0
		for (var j = 0; j < b.length; j++) {
			g += w[k][j] * dS(rA[j]) * dC(rA[j], e[j])
		}
		nextA[k] -= g * eta
	})

	if (l > 0) backprop(a, nextA, weights, biases, l - 1, eta)
}

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

const trainANN = (w, b, trainingSet, testingSet, epochs) => {
	const emptyOut = Array.from({ length: b[b.length - 1].length }, () => 0)
	var eta = 0.08

	for (var i = 1; i <= epochs; i++) {
		const start = Date.now()
		console.log('Epoch ' + i + '...')

		// train
		utils.shuffle(trainingSet)
		trainingSet.forEach(ex => {
			const correct = Object.keys(ex)[0]
			const output = feedForward(w, b, ex[correct])

			var expected = emptyOut.slice()
			expected[correct] = 1

			backprop(output, expected, w, b, b.length - 1, eta)
		})

		const end = Date.now()

		// test
		utils.shuffle(testingSet)
		const testResult = testANN(w, b, testingSet)

		// update learnrate
		eta = testResult.avgCost / 3

		// log
		console.log(
			'Epoch ' +
				i +
				': ' +
				((end - start) / 1000).toFixed(2) +
				's\t' +
				((testResult.correct / testingSet.length) * 100).toFixed(2) +
				'%\t' +
				testResult.avgCost.toFixed(5) +
				'\n'
		)
	}
}

const testANN = (w, b, testingSet) => {
	const emptyOut = Array.from({ length: b[b.length - 1].length }, () => 0)
	var totalCost = 0
	var numCorrect = 0

	testingSet.forEach(ex => {
		const correct = Object.keys(ex)[0]
		const output = feedForward(w, b, ex[correct])

		const result = maxActivation(output[output.length - 1])
		if (correct == result) numCorrect++

		var expected = emptyOut.slice()
		expected[correct] = 1
		const cost = calcCost(expected, output[output.length - 1])

		totalCost += cost
	})

	return {
		avgCost: totalCost / testingSet.length,
		correct: numCorrect
	}
}

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
				column.push(utils.random())
			}
			weightMatrix.push(column)
		}

		for (var j = 0; j < layers[i + 1]; j++) {
			biasVector.push(utils.random())
		}

		weights.push(weightMatrix)
		biases.push(biasVector)
	}

	return {
		weights: weights,
		biases: biases,
		feedForward: function(a) {
			return feedForward(this.weights, this.biases, a)
		},
		calculate: function(a) {
			const res = feedForward(this.weights, this.biases, a)
			return maxActivation(res[res.length - 1])
		},
		train: function(trainingSet, epochs, testingSet) {
			trainANN(this.weights, this.biases, trainingSet, testingSet, epochs)
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
