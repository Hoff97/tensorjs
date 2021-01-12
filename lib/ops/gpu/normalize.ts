import { DrawCommand } from "regl";
import { GPUTensor } from "../../tensor/gpu/tensor";
import { buildComp, compute, copyPad, defaultMain, initIndex, maxRank, pad } from "./util";

let comp: DrawCommand;

const fragmentShader = `
uniform float epsilon;

float process(int index[${maxRank}]) {
  float result = _x(index) - _mean(index);
  result = result / sqrt(_variance(index) + epsilon);
  result = result * _scale(index) + _bias(index);
  return result;
}

${defaultMain}
`;

function initComp() {
  comp = buildComp(['x', 'mean', 'variance', 'scale', 'bias'], fragmentShader, [{name: 'epsilon'}]);
}

export function normalize(x: GPUTensor, mean: GPUTensor, variance: GPUTensor, epsilon: number, scale: GPUTensor, bias: GPUTensor) {
  if (comp === undefined) {
    initComp();
  }

  const resultShape = [...x.shape];

  return compute(comp, resultShape, {
    x, mean, variance, scale, bias
  }, {
    epsilon
  });
}