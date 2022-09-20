// import { logger, sendSuccess, sendError } from "~/utils";
const execSync = require("child_process").execSync;

const model = async (request, response) => {
  const { url, args } = request.body;

  try {
    console.log(url, args);
    const var1 = "./dogs.abc";
    const var2 = "1 2 3 4";
    const output = execSync(
      `python ./scripts/test_onnx_python.py --img_path ${
        url ? url : var1
      } --points "${args ? args : var2}"`,
      { encoding: "utf-8" }
    );
    console.log("Output was:\n", output);

    return response.send({ output });
  } catch (exception) {
    //  Log in case of any abnormal crash
    // logger("error", "Error:", exception.message);
    console.log("Internal Server Error", exception);
  }
};

module.exports = {
  model,
};
