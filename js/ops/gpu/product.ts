import { DrawCommand } from "regl";
import GPUTensor from "../../tensor/gpu/tensor";
import { fragmentShader, initComputation, performComputation } from './pool';

let comp: DrawCommand;

const fragShader = fragmentShader((a, b) => `${a} * ${b}`);

function initComp() {
  comp = initComputation(fragShader);
}

export function product(tensor1: GPUTensor, axes: number[]) {
  if (comp === undefined) {
    initComp();
  }

  return performComputation(tensor1, axes, comp);
}