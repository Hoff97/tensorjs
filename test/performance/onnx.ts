import { OnnxModel } from "../../lib/onnx/model";
import { CPUTensor } from "../../lib/tensor/cpu/tensor";
import Tensor from "../../lib/types";
import { toGPU } from "../../lib/util/convert";

declare const suite: any;
declare const benchmark: any;

const b: ArrayBuffer = require('arraybuffer-loader!../data/mobile_cnn.onnx');
let model: OnnxModel;
let input: Tensor;

suite("Onnx Mobilenet", () => {
  benchmark("On gpu", () => {
    const result = model.forward([input]);

    const sM = result[0].softmax(1);
  });
}, {
  async onStart() {
    model = new OnnxModel(b);
    await model.toGPU();
    input = await toGPU(new CPUTensor([1,3,224,224]));
  }
});