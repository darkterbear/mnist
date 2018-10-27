const Utils = require('./utils')
const fs = require('fs')

const feedForward = (input, w, b) => {
	var activations = [input]

	for (var i = 0; i < b.length; i++) {
		const left = activations[i]

		var right = []

		for (var r = 0; r < w[i][0].length; r++) {
			var sum = 0
			for (var c = 0; c < w[i].length; c++) {
				sum += w[i][c][r] * left[c]
			}
			right.push(Utils.sigmoid(sum + b[i][r]))
		}
		activations.push(right)
	}

	return activations
}

const backprop = (i, w, b, gradW, gradB) => {
	const correct = Object.keys(i)[0]
	const input = i[correct]
	// feedforward
	const result = feedForward(input, w, b)
	// expected
	var expected = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	expected[correct] = 1

	// backwards-pass
	for (var l = b.length - 1; l >= 0; l--) {
		const aL = result[l] // left activations
		const aR = result[l + 1] // right activations

		// console.log(result)

		w[l].forEach((wC, k) => {
			wC.forEach((weight, j) => {
				// tweak gradient by precalculated formula (partial derivativesssss)
				gradW[l][k][j] -=
					aL[k] * aR[j] * (1 - aR[j]) * 2 * (aR[j] - expected[j])
			})
		})

		b[l].forEach((b, j) => {
			gradB[l][j] -= aR[j] * (1 - aR[j]) * 2 * (aR[j] - expected[j])
		})
		var nextExpected = aL.slice()
		aL.forEach((a, k) => {
			var partial = 0
			for (var j = 0; j < b[l].length; j++) {
				partial += w[l][k][j] * aR[j] * (1 - aR[j]) * 2 * (aR[j] - expected[j])
			}
			nextExpected[k] -= partial
		})
		expected = nextExpected
	}
}

const updateMiniBatch = (mb, eta, w, b) => {
	// initialize empty gradients
	var gradW = Array.from({ length: w.length }, (_, l) => {
		return Array.from({ length: w[l].length }, (_, c) => {
			return Array.from({ length: w[l][c].length }, () => 0)
		})
	})
	var gradB = Array.from({ length: b.length }, (_, l) => {
		return Array.from({ length: b[l].length }, () => 0)
	})

	// nudge gradients
	mb.forEach(i => {
		backprop(i, w, b, gradW, gradB)
	})

	// update weights and biases
	for (var l = 0; l < w.length; l++) {
		for (var c = 0; c < w[l].length; c++) {
			for (var r = 0; r < w[l][c].length; r++)
				w[l][c][r] -= (gradW[l][c][r] * eta) / mb.length
		}

		for (var j = 0; j < b[l].length; j++)
			b[l][j] -= (gradB[l][j] * eta) / mb.length
	}
}

const updateSingle = (e, eta, w, b) => {}

const train = (trainSet, epochs, mbSize, eta, testSet, w, b) => {
	for (var e = 1; e <= epochs; e++) {
		console.log('Epoch ' + e + '...')
		const now = Date.now()

		// shuffle trainset
		Utils.shuffle(trainSet)

		// separate into minibatches
		// for (var i = 0; i < trainSet.length; i += mbSize) {
		// 	updateMiniBatch(trainSet.slice(i, i + mbSize), eta, w, b)
		// }

		trainSet.forEach(e => {})

		// test against testset
		const results = test(testSet, w, b)

		console.log(
			(Date.now() - now) / 1000 +
				's; ' +
				(results.correct / testSet.length) * 100 +
				'%\n'
		)
	}
}

const maxActivation = a => {
	return a.reduce((p, c, i) => (c > a[p] ? i : p), 0)
}

const test = (testSet, w, b) => {
	var correct = 0
	for (var i of testSet) {
		for (var expected in i) {
			const a = feedForward(i[expected], w, b)

			if (maxActivation(a[a.length - 1]) == expected) correct++
		}
	}
	return {
		correct
	}
}

const calculate = (input, w, b) => {
	const activations = feedForward(input, w, b)

	return maxActivation(activations[activations.length - 1])
}

const save = (nn, name) => {
	console.log(nn)
	fs.writeFile(
		__dirname + '/models/' + name + '.json',
		JSON.stringify(nn),
		function(err) {
			if (err) {
				return console.log(err)
			}
			console.log('Model Saved')
		}
	)
}

const create = layers => {
	const weights = Array.from({ length: layers.length - 1 }, (_, iM) => {
		const neuronsLeft = layers[iM]
		const neuronsRight = layers[iM + 1]

		// for each matrix, the first dimension is the COLUMNS, each representing
		// the weights for one neuron on the LEFT

		return Array.from({ length: neuronsLeft }, (_, iC) => {
			return Array.from({ length: neuronsRight }, () => Utils.gaussianRand())
		})
	})

	const biases = Array.from({ length: layers.length - 1 }, (_, iL) => {
		return Array.from({ length: layers[iL + 1] }, () => Utils.gaussianRand())
	})

	return {
		weights,
		biases,
		save: name => {
			save({ weights, biases }, name)
		},
		feedForward: inputs => {
			return feedForward(inputs, weights, biases)
		},
		calculate: inputs => {
			return calculate(inputs, weights, biases)
		},
		train: (trainSet, epochs, mbSize, eta, testSet) => {
			return train(trainSet, epochs, mbSize, eta, testSet, weights, biases)
		},
		test: testSet => {
			return test(testSet, weights, biases)
		}
	}
}

const open = filepath => {
	const raw = fs.readFileSync(filepath, 'utf8')

	const obj = JSON.parse(raw)

	return {
		weights: obj.weights,
		biases: obj.biases,
		save: name => {
			save(this, name)
		},
		feedForward: inputs => {
			return feedForward(inputs, obj.weights, obj.biases)
		},
		calculate: inputs => {
			return calculate(inputs, obj.weights, obj.biases)
		},
		train: (trainSet, epochs, mbSize, eta, testSet) => {
			return train(
				trainSet,
				epochs,
				mbSize,
				eta,
				testSet,
				obj.weights,
				obj.biases
			)
		},
		test: testSet => {
			return test(testSet, obj.weights, obj.biases)
		}
	}
}

module.exports = {
	create,
	open
}
