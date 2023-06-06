const express = require("express");
const faceapi = require("face-api.js");
const bodyParser = require("body-parser");
const multer = require("multer");
const app = express();
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
app.use(bodyParser.json());

var storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, "./uploads");
  },
  filename: function (req, file, cb) {
    cb(null, file.originalname);
  },
});
var upload = multer({ storage: storage });

app.post("/face_detection", upload.single("image"), async (req, res) => {
  try {
    if (req.file) {
      var SERVER_URL = req.protocol + "://" + req.get("host");

      const image = req.file.path;

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
          imageURL: SERVER_URL + "/out/faceDetection.png",
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

app.post(
  "/face_expression_recognition",
  upload.single("image"),
  async (req, res) => {
    try {
      if (req.file) {
        var SERVER_URL = req.protocol + "://" + req.get("host");
        const image = req.file.path;
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
            imageURL: SERVER_URL + "/out/faceExpressionRecognition.png",
            detections: detections.length
              ? detections.map((e) => e["detection"])
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

app.post("/face_recognition", upload.array("images", 2), async (req, res) => {
  try {
    if (req.files && req.files.length > 1) {
      var SERVER_URL = req.protocol + "://" + req.get("host");

      const REFERENCE_IMAGE = req.files[0].path;
      const QUERY_IMAGE = req.files[1].path;

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

      res.json({
        success: {
          imageURL: SERVER_URL + "/out/referenceImage.png",
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
