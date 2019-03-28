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
    this.outputCanvas = document.querySelector('#outputCanvas');
    this.outputCanvas.width = IMAGE_SIZE;
    this.outputCanvas.height = IMAGE_SIZE;

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
    this.model = await tf.loadLayersModel('/fast_tfjs_model/model.json');
    this.model_ready = true;
  }

  async predict() {
    let red, green, blue;
    let colour_channel_shape = [IMAGE_SIZE, IMAGE_SIZE, 1];

    let input = tf.browser.fromPixels(this.inputCanvas, 3).asType('float32');

    red = tf.add(input.slice([0, 0, 0], colour_channel_shape), tf.fill(colour_channel_shape, -103.939));
    green = tf.add(input.slice([0, 0, 1], colour_channel_shape), tf.fill(colour_channel_shape, -116.779));
    blue = tf.add(input.slice([0, 0, 2], colour_channel_shape), tf.fill(colour_channel_shape, -123.68));
    input = tf.concat([red, green, blue], 2)

    let output = await this.model.predict(input.expandDims(0)); // tf.reshape(input), [1, IMAGE_SIZE, IMAGE_SIZE, 3]));
    output = output.squeeze(0);
    red = tf.add(output.slice([0, 0, 0], colour_channel_shape), tf.fill(colour_channel_shape, 103.939));
    green = tf.add(output.slice([0, 0, 1], colour_channel_shape), tf.fill(colour_channel_shape, 116.779));
    blue = tf.add(output.slice([0, 0, 2], colour_channel_shape), tf.fill(colour_channel_shape, 123.68));
    output = tf.concat([red, green, blue], 2).asType('int32');
    output = tf.clipByValue(output, 0, 255);
    tf.browser.toPixels(output, this.outputCanvas);
    this.predicting = false;
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
