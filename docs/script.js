window.onload = () => {
	video = document.querySelector('video');
	navigator.mediaDevices.getUserMedia({
		video: {
			facingMode: "user"
		}
	}).then(stream => video.srcObject = stream)
	.catch(error => alert("Error while trying to connect to camera"));

}
