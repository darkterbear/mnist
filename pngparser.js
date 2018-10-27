const getPixels = require('get-pixels')

const parsePNG = (filepath, cb) => {
	getPixels(filepath, (err, pixels) => {
		if (err) {
			console.log('Bad image path')
			return
		}

		var brightnesses = []

		for (var i = 0; i < pixels.data.length; i += 4) {
			brightnesses.push(
				Math.round(
					(pixels.data[i] + pixels.data[i + 1] + pixels.data[i + 2]) / 3
				)
			)
		}
		cb(brightnesses)
	})
}

module.exports = parsePNG
