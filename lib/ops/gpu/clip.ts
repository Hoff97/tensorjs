import { DrawCommand } from "regl";
import { GPUTensor } from "../../tensor/gpu/tensor";
import { buildComp, compute } from "./util";

let compMinMax: DrawCommand;
let compMin: DrawCommand;
let compMax: DrawCommand;

const fragmentShaderMinMax = `
uniform float minVal;
uniform float maxVal;

void main() {
  vec4 maxVec = vec4(maxVal,maxVal,maxVal,maxVal);
  vec4 minVec = vec4(minVal,minVal,minVal,minVal);

  gl_FragColor = min(maxVec, max(minVec, texture2D(inputTensor, uv)));
}`;

const fragmentShaderMin = `
uniform float minVal;

void main() {
  vec4 minVec = vec4(minVal,minVal,minVal,minVal);

  gl_FragColor = max(minVec, texture2D(inputTensor, uv));
}`;

const fragmentShaderMax = `
uniform float maxVal;

void main() {
  vec4 maxVec = vec4(maxVal,maxVal,maxVal,maxVal);

  gl_FragColor = min(maxVec, texture2D(inputTensor, uv));
}`;

function initComp() {
  compMinMax = buildComp(['inputTensor'], fragmentShaderMinMax, [{name: 'minVal'}, {name: 'maxVal'}]);
  compMin = buildComp(['inputTensor'], fragmentShaderMin, [{name: 'minVal'}]);
  compMax = buildComp(['inputTensor'], fragmentShaderMax, [{name: 'maxVal'}]);
}

export function clip(tensor: GPUTensor, min?: number, max?: number) {
  if (compMinMax === undefined) {
    initComp();
  }

  if (min !== undefined && max !== undefined) {
    return compute(compMinMax, tensor.getShape(), {
      inputTensor: tensor
    }, {minVal: min, maxVal: max});
  } else if (min !== undefined) {
    return compute(compMin, tensor.getShape(), {
      inputTensor: tensor
    }, {minVal: min});
  } else if (max !== undefined) {
    return compute(compMax, tensor.getShape(), {
      inputTensor: tensor
    }, {maxVal: max});
  }
  return tensor.copy();
}