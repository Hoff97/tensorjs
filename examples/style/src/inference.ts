import * as tjs from '@hoff97/tensor-js';


export async function loadModel() {
    const res = await fetch('mosaic.onnx');
    const buffer = await res.arrayBuffer();

    const model = new tjs.onnx.model.OnnxModel(buffer);
    await model.toGPU();

    return model;
}