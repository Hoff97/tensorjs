import { DrawCommand } from "regl";
import GPUTensor from "../../tensor/gpu/tensor";
import { buildComp, compute } from "./util";

let comp: DrawCommand;

const fragmentShader = `
precision mediump float;
uniform sampler2D inputTensor1;
uniform sampler2D inputTensor2;
varying vec2 uv;
void main() {
  gl_FragColor = texture2D(inputTensor1, uv) / texture2D(inputTensor2, uv);
}`;

function initComp() {
  comp = buildComp(['inputTensor1', 'inputTensor2'], fragmentShader);
}

export function divide(tensor1: GPUTensor, tensor2: GPUTensor) {
  if (comp === undefined) {
    initComp();
  }

  return compute(comp, tensor1.getShape(), {
    inputTensor1: tensor1,
    inputTensor2: tensor2
  });
}