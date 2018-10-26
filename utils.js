const gaussianRand = () => {
	return (
		Array.from({ length: 4 }, () => Math.random() - 0.5).reduce(
			(a, c) => a + c
		) / 2
	)
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
	return Array.from({ length: v1.length }, (_, i) => v1[i] + v2[i])
}

const mVMult = (m, v) => {
	return Array.from({ length: m[0].length }, (_, r) => {
		return Array.from({ length: v.length }, (_, c) => m[c][r] * v[c]).reduce(
			(a, c) => a + c
		)
	})
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
