import { DrawCommand } from "regl";
import GPUTensor from "../../tensor/gpu/tensor";
import { buildComp, compute, defaultMain, maxRank } from "./util";

let comp: DrawCommand;

const fragmentShader = `
uniform int axis;

float process(int index[${maxRank}]) {
  float res = 0.0;
  for (int i = 0; i < ${maxRank}; i++) {
    if (i == axis) {
      if (index[i] >= shapeinputTensor1[i]) {
        index[i] = index[i] - shapeinputTensor1[i];
        res += _inputTensor2(index);
      } else {
        res += _inputTensor1(index);
      }
      break;
    }
  }
  return res;
}

${defaultMain}
`;

function initComp() {
  comp = buildComp(['inputTensor1', 'inputTensor2'], fragmentShader, [{name: 'axis'}]);
}

export function concat(tensor1: GPUTensor, tensor2: GPUTensor, axis: number) {
  if (comp === undefined) {
    initComp();
  }

  const outputShape = [...tensor1.shape];
  outputShape[axis] += tensor2.shape[axis];

  return compute(comp, outputShape, {
    inputTensor1: tensor1,
    inputTensor2: tensor2
  }, {
    axis
  });
}