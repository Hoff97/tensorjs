import { DrawCommand } from "regl";
import { GPUTensor } from "../../tensor/gpu/tensor";
import { buildComp, compute } from "./util";

let comp: DrawCommand;

const fragmentShader = `
void main() {
  gl_FragColor = texture2D(inputTensor, uv);
}`;

function initComp() {
  comp = buildComp(['inputTensor'], fragmentShader);
}

export function copy(tensor: GPUTensor, newShape?: number[]) {
  if (comp === undefined) {
    initComp();
  }

  if (newShape === undefined) {
    newShape = [...tensor.getShape()];
  }

  return compute(comp, newShape, {
    inputTensor: tensor
  });
}