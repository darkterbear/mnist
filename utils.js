const spread = 1.125
const random = () => {
	return Math.random() * spread - spread / 2
}

const shuffle = a => {
	for (let i = a.length - 1; i > 0; i--) {
		const j = Math.floor(Math.random() * (i + 1))
		;[a[i], a[j]] = [a[j], a[i]]
	}
	return a
}

const sigmoid = z => {
	return 1 / (1 + Math.E ** -z)
}

const dSigmoid = s => {
	return s * (1 - s)
}

const dCost = (a, e) => {
	return 2 * (a - e)
}

module.exports = {
	shuffle,
	sigmoid,
	dCost,
	dSigmoid,
	random
}
