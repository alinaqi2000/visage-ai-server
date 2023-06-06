import * as fs from "fs";
import { promisify } from "util";

const unlinkAsync = promisify(fs.unlink);

export async function removeFile(fileName: string) {
  unlinkAsync(fileName);
}
