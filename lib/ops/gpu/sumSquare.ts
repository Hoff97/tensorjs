import { DrawCommand } from "regl";
import { GPUTensor } from "../../tensor/gpu/tensor";
import { fragmentShader, initComputation, performComputation } from './pool';

let comp: DrawCommand;

const fragShader = fragmentShader((a, b) => `(${a}*${a}) + ${b}`, undefined, (res) => `${res}*${res}`);

function initComp() {
  comp = initComputation(fragShader);
}

export function sumSquare(tensor1: GPUTensor, axes: number[], keepDims: boolean) {
  if (comp === undefined) {
    initComp();
  }

  return performComputation(tensor1, axes, keepDims, comp);
}