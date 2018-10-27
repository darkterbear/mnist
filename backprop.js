const utils = require('./utils')

const dS = utils.dSigmoid
const dC = utils.dCost

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
