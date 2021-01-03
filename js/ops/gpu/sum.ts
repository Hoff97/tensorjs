import { DrawCommand } from "regl";
import GPUTensor from "../../tensor/gpu/tensor";
import { fragmentShader, initComputation, performComputation } from './pool';

let comp: DrawCommand;

const fragShader = fragmentShader((a, b) => `${a} + ${b}`);

function initComp() {
  comp = initComputation(fragShader);
}

export function sum(tensor1: GPUTensor, axes: number[], dest?: GPUTensor) {
  if (comp === undefined) {
    initComp();
  }

  return performComputation(tensor1, axes, comp, dest);
}