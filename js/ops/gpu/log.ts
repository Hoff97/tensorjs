import { DrawCommand } from "regl";
import GPUTensor from "../../tensor/gpu/tensor";
import { buildComp, compute } from "./util";

let comp: DrawCommand;

const fragmentShader = `
precision mediump float;
uniform sampler2D inputTensor;
varying vec2 uv;
void main() {
  gl_FragColor = log(texture2D(inputTensor, uv));
}`;

function initComp() {
  comp = buildComp(['inputTensor'], fragmentShader);
}

export function log(tensor: GPUTensor) {
  if (comp === undefined) {
    initComp();
  }

  return compute(comp, tensor.getShape(), {
    inputTensor: tensor
  });
}