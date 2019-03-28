const IMAGE_SIZE = 256;

class App {
  constructor() {
    this.model_ready = false;
    this.prediction_ready = false;
    this.predicting = false;
    this.model = {};
    this.loadModel()
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

  async loadModel() {
    this.model = await tf.loadLayersModel('/tfjs_model/model.json');
    this.model_ready = true;
  }

  async predict() {
    let input = tf.browser.fromPixels(this.inputCanvas, 3);
    // input.toFloat().print();
    let output = await this.model.predict(tf.reshape(input.toFloat(), [1, IMAGE_SIZE, IMAGE_SIZE, 3]));
  }

  drawInputVideo() {
    let videoWidth = this.inputVideo.videoWidth;
    let videoHeight = this.inputVideo.videoHeight;
    let minVideoLength = Math.min(videoWidth, videoHeight)
    
    if (!this.predicting && this.model_ready) {
      this.predicting = true;
      this.predict();
    }

    // Top left pixel of source video we wish to draw
    let sx = 0;
    let sy = 0; 

    if (videoWidth > videoHeight) {
      sx = (videoWidth - videoHeight)/2;
    } else {
      sy = (videoHeight - videoWidth)/2;
    }
    this.inputCtx.drawImage(this.inputVideo, sx, sy, minVideoLength, minVideoLength, 0, 0, IMAGE_SIZE, IMAGE_SIZE);
    window.requestAnimationFrame(() => this.drawInputVideo());
  }
}

window.onload = () => {window.app = new App()};
