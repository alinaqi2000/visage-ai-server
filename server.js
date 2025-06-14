const express = require("express");
const cors = require("cors");
const path = require("path");
const fs = require("fs");
const faceapi = require("face-api.js");
const { Canvas, Image, ImageData } = require("canvas");
const multer = require("multer");
const tf = require("@tensorflow/tfjs");
require("@tensorflow/tfjs-node");

// Initialize TensorFlow.js with CPU backend
const tfjsBackend = tf.backend();
if (tfjsBackend.name !== 'cpu') {
  tf.setBackend('cpu');
}

const bodyParser = require("body-parser");

const app = express();
app.use(
  cors({
    origin: "*",
  })
);
const PORT = 3000;
const {
  canvas,
  faceDetectionNet,
  faceDetectionOptions,
  saveFile,
  removeFile,
} = require("./commons");

app.use("/images", express.static(__dirname + "/images"));
app.use("/out", express.static(__dirname + "/out"));

// app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(bodyParser.json({ limit: "50mb" }));

var storage = multer.diskStorage({
  destination: function (req, file, cb) {
    var baseDir = path.resolve(__dirname, "./uploads");

    if (!fs.existsSync(baseDir)) {
      fs.mkdirSync(baseDir);
    }

    cb(null, "./uploads");
  },
  filename: function (req, file, cb) {
    cb(null, file.originalname);
  },
});
var upload = multer();

async function uploadBase64(req, name) {
  var base64Data = req.body[name].data;
  var file_name = req.body[name].name;
  let base64Image = base64Data.split(";base64,").pop();
  return new Promise(async (resolve, reject) => {
    await fs.writeFile(
      __dirname + "/uploads/" + file_name,
      base64Image,
      "base64",
      function (err) {
        if (err) console.log(err);
        resolve("done");
      }
    );
    resolve("done");
  });
}
app.get("/", (req, res) => {
  res.json({ "API": "Morphous AI Server", "version": "1.0.0" });
});

app.post("/face_detection", upload.array(), async (req, res) => {
  try {
    var baseDir = path.resolve(__dirname, "./uploads");

    if (!fs.existsSync(baseDir)) {
      fs.mkdirSync(baseDir);
    }
    var file_name = req.body.image.name;
    if (file_name) {
      await uploadBase64(req, "image");
      const image = __dirname + "/uploads/" + file_name;
      var SERVER_URL = req.protocol + "://" + req.get("host");
      await faceDetectionNet.loadFromDisk("./weights");
      const img = await canvas.loadImage(image);
      await removeFile(image);
      const detections = await faceapi.detectAllFaces(
        img,
        faceDetectionOptions
      );
      const out = faceapi.createCanvasFromMedia(img);
      faceapi.draw.drawDetections(out, detections);
      saveFile("faceDetection.png", out.toBuffer("image/png"));
      res.json({
        success: {
          imageData: out.toDataURL(),
          imageURL:
            SERVER_URL +
            "/out/faceDetection.png?" +
            Math.round(Math.random() * 10000000000),
          detections: detections,
        },
      });
    } else {
      res.json({
        error: "Please add a valid image.",
      });
    }
  } catch (e) {
    res.json({
      error: e,
    });
  }
});

app.post("/age_and_gender_recognition", upload.array(), async (req, res) => {
  try {
    var baseDir = path.resolve(__dirname, "./uploads");

    if (!fs.existsSync(baseDir)) {
      fs.mkdirSync(baseDir);
    }
    var file_name = req.body.image.name;
    if (file_name) {
      await uploadBase64(req, "image");
      const image = __dirname + "/uploads/" + file_name;
      var SERVER_URL = req.protocol + "://" + req.get("host");

      await faceDetectionNet.loadFromDisk("./weights");
      await faceapi.nets.faceLandmark68Net.loadFromDisk("./weights");
      await faceapi.nets.ageGenderNet.loadFromDisk("./weights");

      const img = await canvas.loadImage(image);

      await removeFile(image);
      const detections = await faceapi
        .detectAllFaces(img, faceDetectionOptions)
        .withFaceLandmarks()
        .withAgeAndGender();

      const out = faceapi.createCanvasFromMedia(img);
      faceapi.draw.drawDetections(
        out,
        detections.map((res) => res.detection)
      );
      detections.forEach((result) => {
        const { age, gender, genderProbability } = result;
        new faceapi.draw.DrawTextField(
          [
            `${faceapi.utils.round(age, 0)} years`,
            `${gender} (${faceapi.utils.round(genderProbability)})`,
          ],
          result.detection.box.bottomLeft
        ).draw(out);
      });

      saveFile("faceDetection.png", out.toBuffer("image/png"));
      res.json({
        success: {
          imageData: out.toDataURL(),
          imageURL:
            SERVER_URL +
            "/out/faceDetection.png?" +
            Math.round(Math.random() * 10000000000),
          detections: detections.length
            ? detections.map((e) => ({
              ...e["detection"],
              gender: e.gender,
              genderProbability: e.genderProbability,
              age: e.age,
            }))
            : [],
        },
      });
    } else {
      res.json({
        error: "Please add a valid image.",
      });
    }
  } catch (e) {
    res.json({
      error: e,
    });
  }
});

app.post(
  "/face_expression_recognition",
  upload.single("image"),
  async (req, res) => {
    try {
      var baseDir = path.resolve(__dirname, "./uploads");

      if (!fs.existsSync(baseDir)) {
        fs.mkdirSync(baseDir);
      }
      var file_name = req.body.image.name;
      if (file_name) {
        await uploadBase64(req, "image");
        const image = __dirname + "/uploads/" + file_name;

        var SERVER_URL = req.protocol + "://" + req.get("host");
        await faceDetectionNet.loadFromDisk("./weights");
        await faceapi.nets.faceLandmark68Net.loadFromDisk("./weights");
        await faceapi.nets.faceExpressionNet.loadFromDisk("./weights");

        const img = await canvas.loadImage(image);
        const detections = await faceapi
          .detectAllFaces(img, faceDetectionOptions)
          .withFaceLandmarks()
          .withFaceExpressions();

        await removeFile(image);

        const out = faceapi.createCanvasFromMedia(img);
        faceapi.draw.drawDetections(
          out,
          detections.map((res) => res.detection)
        );
        faceapi.draw.drawFaceExpressions(out, detections);

        saveFile("faceExpressionRecognition.png", out.toBuffer("image/png"));
        res.json({
          success: {
            imageData: out.toDataURL(),
            imageURL:
              SERVER_URL +
              "/out/faceExpressionRecognition.png?" +
              Math.round(Math.random() * 10000000000),
            detections: detections.length
              ? detections.map((e) => ({
                ...e["detection"],
                expressions: { ...e["expressions"] },
              }))
              : [],
          },
        });
      } else {
        res.json({
          error: "Please add a valid image.",
        });
      }
    } catch (e) {
      res.json({
        error: e,
      });
    }
  }
);

app.post("/face_recognition", upload.array(), async (req, res) => {
  try {
    var baseDir = path.resolve(__dirname, "./uploads");

    if (!fs.existsSync(baseDir)) {
      fs.mkdirSync(baseDir);
    }
    var file_name = req.body.image.name;
    var query_name = req.body.query.name;
    if (file_name) {
      await uploadBase64(req, "image");
      await uploadBase64(req, "query");
      const REFERENCE_IMAGE = __dirname + "/uploads/" + file_name;
      const QUERY_IMAGE = __dirname + "/uploads/" + query_name;

      var SERVER_URL = req.protocol + "://" + req.get("host");

      await faceDetectionNet.loadFromDisk("./weights");
      await faceapi.nets.faceLandmark68Net.loadFromDisk("./weights");
      await faceapi.nets.faceRecognitionNet.loadFromDisk("./weights");

      const referenceImage = await canvas.loadImage(REFERENCE_IMAGE);
      const queryImage = await canvas.loadImage(QUERY_IMAGE);

      const resultsRef = await faceapi
        .detectAllFaces(referenceImage, faceDetectionOptions)
        .withFaceLandmarks()
        .withFaceDescriptors();

      await removeFile(REFERENCE_IMAGE);
      await removeFile(QUERY_IMAGE);

      const resultsQuery = await faceapi
        .detectAllFaces(queryImage, faceDetectionOptions)
        .withFaceLandmarks()
        .withFaceDescriptors();

      const faceMatcher = new faceapi.FaceMatcher(resultsRef);

      const labels = faceMatcher.labeledDescriptors.map((ld) => ld.label);
      const refDrawBoxes = resultsRef
        .map((res) => res.detection.box)
        .map((box, i) => new faceapi.draw.DrawBox(box, { label: labels[i] }));
      const outRef = faceapi.createCanvasFromMedia(referenceImage);
      refDrawBoxes.forEach((drawBox) => drawBox.draw(outRef));

      saveFile("referenceImage.png", outRef.toBuffer("image/png"));

      const queryDrawBoxes = resultsQuery.map((res) => {
        const bestMatch = faceMatcher.findBestMatch(res.descriptor);
        return new faceapi.draw.DrawBox(res.detection.box, {
          label: bestMatch.toString(),
        });
      });
      const outQuery = faceapi.createCanvasFromMedia(queryImage);
      queryDrawBoxes.forEach((drawBox) => drawBox.draw(outQuery));

      saveFile("queryImage.png", outQuery.toBuffer("image/png"));

      res.json({
        success: {
          imageData: outRef.toDataURL(),
          queryData: outQuery.toDataURL(),
          queryURL:
            SERVER_URL +
            "/out/queryImage.png?" +
            Math.round(Math.random() * 10000000000),
          imageURL:
            SERVER_URL +
            "/out/referenceImage.png?" +
            Math.round(Math.random() * 10000000000),
          detections: resultsQuery.length
            ? resultsQuery.map((e) => e["detection"])
            : [],
        },
      });
    } else {
      res.json({
        error: "Please add valid images.",
      });
      s.length > 1;
    }
  } catch (e) {
    res.json({
      error: e,
    });
  }
});

app.listen(PORT, () => console.log(`Listening on port ${PORT}!`));
