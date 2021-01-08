import { DrawCommand } from "regl";
import GPUTensor from "../../tensor/gpu/tensor";
import { buildComp, compute, defaultMain, maxRank } from "./util";

let comp: DrawCommand;

const fragmentShader = `
float process(int index[${maxRank}]) {
  return _inputTensor1(index) / _inputTensor2(index);
}

${defaultMain}
`;

function initComp() {
  comp = buildComp(['inputTensor1', 'inputTensor2'], fragmentShader);
}

export function divide(tensor1: GPUTensor, tensor2: GPUTensor, resultShape: readonly number[]) {
  if (comp === undefined) {
    initComp();
  }

  return compute(comp, resultShape, {
    inputTensor1: tensor1,
    inputTensor2: tensor2
  });
}