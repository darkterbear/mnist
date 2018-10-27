const utils = require('./utils')

// TODO: don't use these, slows down by about a second by repeated bloating callstack
const dS = utils.dSigmoid
const dC = utils.dCost

/**
 *
 * @param {Matrix} a Activations of the entire neural network
 * @param {Vector} e Expected activations of this current layer
 * @param {[Matrix]} weights Weights of the network
 * @param {[Vector]} biases Biases of the network
 * @param {Number} l Layer of the network
 * @param {Number} eta Learning rate
 */
const backprop = (a, e, weights, biases, l, eta) => {
	var w = weights[l - 1]
	var b = biases[l - 1]

	var rA = a[l]
	var lA = a[l - 1]

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

	if (l > 1) backprop(a, nextA, weights, biases, l - 1, eta)
}

module.exports = backprop
