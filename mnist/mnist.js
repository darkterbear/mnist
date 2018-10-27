const fs = require('fs')
const utils = require('../utils')

const parseFile = (images, labels, numImages) => {
	var dataFileBuffer = fs.readFileSync(__dirname + images)
	var labelFileBuffer = fs.readFileSync(__dirname + labels)
	var pixelValues = []

	for (var image = 0; image < numImages; image++) {
		var pixels = []

		for (var x = 0; x < 28; x++) {
			for (var y = 0; y < 28; y++) {
				pixels.push(dataFileBuffer[image * 28 * 28 + (x + y * 28) + 15] / 255)
			}
		}

		var imageData = {}
		imageData[JSON.stringify(labelFileBuffer[image + 8])] = pixels

		pixelValues.push(imageData)
	}

	return pixelValues
}

var trainingPixelValues = parseFile(
	'/train-images.idx3-ubyte',
	'/train-labels.idx1-ubyte',
	60000
)

utils.shuffle(trainingPixelValues)

var testingPixelValues = parseFile(
	'/t10k-images-idx3-ubyte',
	'/t10k-labels-idx1-ubyte',
	10000
)
utils.shuffle(testingPixelValues)

module.exports = {
	trainingSet: trainingPixelValues,
	testSet: testingPixelValues
}
