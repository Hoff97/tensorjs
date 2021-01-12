import { DrawCommand } from "regl";
import { GPUTensor } from "../../tensor/gpu/tensor";
import { buildComp, compute } from "./util";

let comp: DrawCommand;

const fragmentShader = `
void main() {
  gl_FragColor = floor(texture2D(inputTensor, uv));
}`;

function initComp() {
  comp = buildComp(['inputTensor'], fragmentShader);
}

export function floor(tensor: GPUTensor) {
  if (comp === undefined) {
    initComp();
  }

  return compute(comp, tensor.getShape(), {
    inputTensor: tensor
  });
}