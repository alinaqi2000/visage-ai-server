"use strict";
exports.__esModule = true;

var fs = require("fs");
var { promisify } = require("util");
const unlinkAsync = promisify(fs.unlink);

function removeFile(fileName) {
  unlinkAsync(fileName);
}
exports.removeFile = removeFile;
