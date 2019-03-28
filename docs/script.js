const IMAGE_SIZE = 256;
class App {
  constructor() {
    this.inputVideo = document.querySelector('video');
    this.inputCanvas = document.querySelector('#inputCanvas');
    this.inputCanvas.width = IMAGE_SIZE;
    this.inputCanvas.height = IMAGE_SIZE;
    this.inputCtx = this.inputCanvas.getContext('2d');

    navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user"
      }
    }).then(stream => this.inputVideo.srcObject = stream)
    .catch(error => alert("Error while trying to connect to camera"));

    window.requestAnimationFrame(() => this.drawInputVideo());

    console.log(this.inputVideo);
  }
  
  drawInputVideo() {
    let videoWidth = this.inputVideo.videoWidth;
    let videoHeight = this.inputVideo.videoHeight;
    let minVideoLength = Math.min(videoWidth, videoHeight)

    // Top left pixel of source video we wish to draw
    let sx = 0
    let sy = 0; 

    if (videoWidth > videoHeight) {
      sx = (videoWidth - videoHeight)/2;
    } else {
      sy = (videoHeight - videoWidth)/2;
    }
    this.inputCtx.drawImage(this.inputVideo, sx, sy, minVideoLength, minVideoLength, 0, 0, IMAGE_SIZE, IMAGE_SIZE); //this.inputVideo.videoWidth, this.inputVideo.videoHeight);
    t += 1;
    window.requestAnimationFrame(() => this.drawInputVideo());
  }
}
let t = 1


window.onload = () => {window.app = new App()};
