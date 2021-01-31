import * as tjs from '@hoff97/tensor-js';


export async function loadModel(name: string) {
    const res = await fetch(`models/${name}.onnx`);
    const buffer = await res.arrayBuffer();

    const model = new tjs.onnx.model.OnnxModel(buffer, {
        noConvertNodes: [69, 98],
        precision: 16
    });
    await model.toGPU();

    return model;
}