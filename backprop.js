const backprop = (activations, expected, weights, biases, layer, learnRate) => {
	var thisWeights = weights[layer - 1]
	var thisBiases = biases[layer - 1]

	var thisActivations = activations[layer]
	var prevActivations = activations[layer - 1]

	thisWeights.forEach((weightColumn, k) => {
		weightColumn.forEach((weight, j) => {
			const gradient =
				prevActivations[k] *
				(thisActivations[j] * (1 - thisActivations[j])) *
				2 *
				(thisActivations[j] - expected[j])

			weights[layer - 1][k][j] -= gradient * learnRate
		})
	})

	thisBiases.forEach((bias, j) => {
		const gradient =
			thisActivations[j] *
			(1 - thisActivations[j]) *
			2 *
			(thisActivations[j] - expected[j])

		biases[layer - 1][j] -= gradient * learnRate
	})

	var nextLayerExpected = prevActivations.slice()
	prevActivations.forEach((prevActivation, k) => {
		var gradient = 0
		for (var j = 0; j < thisBiases.length; j++) {
			gradient +=
				thisWeights[k][j] *
				(thisActivations[j] * (1 - thisActivations[j])) *
				2 *
				(thisActivations[j] - expected[j])
		}
		nextLayerExpected[k] -= gradient * learnRate
	})

	if (layer > 1)
		backprop(
			activations,
			nextLayerExpected,
			weights,
			biases,
			layer - 1,
			learnRate
		)
}

module.exports = backprop
