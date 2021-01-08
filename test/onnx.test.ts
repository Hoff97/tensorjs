import { CPUTensor } from '../lib/library';
import { OnnxModel } from '../lib/onnx/model';
import { toGPU } from '../lib/util/convert';

const b: ArrayBuffer = require('arraybuffer-loader!./data/mobile_cnn.onnx');

/*describe(`Onnx model`, () => {
  it('should be able to load a MobileNet', async () => {
    const model = new OnnxModel(b);

    await model.toGPU();

    const input = await toGPU(new CPUTensor([1,3,224,224]));
    const result = model.forward([input]);

    const sM = result[0].softmax(1);

    const values = await sM.getValues();
  })
});*/