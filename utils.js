const gaussianRand = () => {
	var u = 0,
		v = 0
	while (u === 0) u = Math.random() //Converting [0,1) to (0,1)
	while (v === 0) v = Math.random()
	return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v)
}

const shuffle = a => {
	for (let i = a.length - 1; i > 0; i--) {
		const j = Math.floor(Math.random() * (i + 1))
		;[a[i], a[j]] = [a[j], a[i]]
	}
}

const sigmoid = z => {
	return 1 / (1 + Math.E ** -z)
}

const sigmoidPrime = z => {
	return sigmoid(z) * (1 - sigmoid(z))
}

const relu = z => {
	return z > 0 ? z : 0
}

const reluPrime = z => {
	return z > 0 ? 1 : 0
}

const vAdd = (v1, v2) => {
	var r = []
	for (var i = 0; i < v1.length; i++) r.push(v1[i] + v2[i])
	return r
}

const mVMult = (m, v) => {
	var res = []

	for (var r = 0; r < m[0].length; r++) {
		var s = 0
		for (var c = 0; c < v.length; c++) {
			s += m[c][r] * v[c]
		}
		res.push(s)
	}
	return res
}

module.exports = {
	gaussianRand,
	shuffle,
	relu,
	reluPrime,
	sigmoid,
	sigmoidPrime,
	vAdd,
	mVMult
}
