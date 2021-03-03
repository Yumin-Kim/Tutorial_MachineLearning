const RtspServer = require('rtsp-streaming-server').default;

const server = new RtspServer({
	serverPort: 5504,
	clientPort: 3434,
	rtpPortStart: 10000,
	rtpPortCount: 10000
});


async function run () {
	try {
		console.log("Start")
		await server.start();
		console.log("Test")
	} catch (e) {
		console.error(e);
	}
}

run();


// const Stream = require('node-rtsp-stream-jsmpeg')

// const options = {
//   name: 'streamName',
//   url: 'rtsp://192.168.219.130:5504/web',
//   wsPort: 3333
// }

// stream = new Stream(options)
// stream.start()