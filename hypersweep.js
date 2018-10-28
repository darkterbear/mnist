/**
 * sweeps all viable hyperparameters and spawns corresponding node processes to test
 */

const { fork } = require('child_process')

for (var iLR = 0.05; iLR <= 0.3; iLR += 0.01) {
	for (var d = -0.05; d >= -0.3; d -= 0.01) {
		const child = fork('./autoeval', [iLR, d])
	}
}
