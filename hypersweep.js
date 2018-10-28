/**
 * sweeps all viable hyperparameters and spawns corresponding node processes to test
 */

const { fork } = require('child_process')

const maxRunning = 12

var configs = []
const iLR = 0.19
for (var d = -0.05; d >= -0.3; d -= 0.01) {
	for (var i = 1; i <= 5; i++) {
		configs.push([iLR, d, i])
	}
}

const handleProcExit = () => {
	if (configs.length > 0) {
		startProc(configs.splice(0, 1)[0])
	}
}

const startProc = config => {
	const child = fork('./autoeval', config, { stdio: 'ignore' })
	child.on('exit', handleProcExit)
}

for (var i = 0; i < maxRunning; i++) {
	startProc(configs.splice(0, 1)[0])
}
