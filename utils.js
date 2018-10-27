const range = 1.125
const random = () => {
	var u = 0,
		v = 0
	while (u === 0) u = Math.random() //Converting [0,1) to (0,1)
	while (v === 0) v = Math.random()
	let num = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v)
	num = num / 10.0 + 0.5 // Translate to 0 -> 1
	if (num > 1 || num < 0) return randn_bm() // resample between 0 and 1
	return num * range - range / 2
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
