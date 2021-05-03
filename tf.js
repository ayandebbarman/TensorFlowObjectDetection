const tf = require("@tensorflow/tfjs-node");
const automl = require("@tensorflow/tfjs-automl");
const fs = require("fs");

const model_url = "/model.json";
const image_path = process.argv.slice(2)[0];

if (!image_path) {
  throw new Error("/MegaFortuneWheel.png");
}

const image = fs.readFileSync("/MegaFortuneWheel.png");
const decoded_image = tf.node.decodeJpeg(image);

async function run() {
  const model = await automl.loadObjectDetection(model_url);
  const predictions = await model.detect(decoded_image);

  console.log(predictions);
}

run().catch(console.error);