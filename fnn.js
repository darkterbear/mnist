const Utils = require('./utils')
const fs = require('fs')

const feedForward = (input, w, b) => {
	var activations = [input]

	for (var i = 0; i < b.length; i++) {
		const left = activations[i]

		activations.push(
			Utils.vAdd(Utils.mVMult(w[i], left), b[i]).map(z => Utils.sigmoid(z))
		)
	}

	return activations
}

const backprop = (i, w, b, gradW, gradB) => {
	const expected = Object.keys(i)[0]
	const input = i[expected]

	// feedforward
	const result = feedForward(input, w, b)

	// expected
	const expectedActivation = Array.from(
		{ length: b[b.length - 1].length },
		() => 0
	)
	expectedActivation[expected] = 1

	// backwards-pass
}

const updateMiniBatch = (mb, eta, w, b) => {
	var gradW = Array.from({ length: w.length }, (_, l) => {
		return Array.from({ length: w[l].length }, (_, c) => {
			return Array.from({ length: w[l][c].length }, () => 0)
		})
	})
	var gradB = Array.from({ length: b.length }, (_, l) => {
		return Array.from({ length: b[l].length }, () => 0)
	})

	mb.forEach(i => {
		backprop(i, w, b)
	})
}

const train = (trainSet, epochs, mbSize, eta, testSet, w, b) => {
	for (var e = 1; e <= epochs; e++) {
		Utils.shuffle(trainSet)

		for (var i = 0; i < trainSet.length; i += mbSize) {
			console.log('here')
			updateMiniBatch(trainSet.slice(i * mbSize, (i + 1) * mbSize), eta, w, b)
		}
	}
}

const test = () => {}

const calculate = (input, w, b) => {
	const activations = feedForward(input, w, b)

	var index = 0,
		max = Number.MIN_SAFE_INTEGER
	activations[activations.length - 1].forEach((a, i) => {
		if (a > max) {
			max = a
			index = i
		}
	})

	return index
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
		}
	}
}

module.exports = {
	create,
	open
}
