import { DrawCommand } from "regl";
import GPUTensor from "../../tensor/gpu/tensor";
import { computeStrides } from "../../util/shape";
import { buildComp, compute, defaultMain, maxRank, pad } from "./util";

let comp: DrawCommand;

const fragmentShader = `
uniform int mappedStrides[${maxRank}];

float process(int index[${maxRank}]) {
  return getValueAt(index, mappedStrides, widthx, heightx, x);
}

${defaultMain}
`;

function initComp() {
  comp = buildComp(['x'], fragmentShader, [{name: 'mappedStrides', length: maxRank}]);
}

export function transpose(tensor: GPUTensor, permutation: number[]) {
  if (comp === undefined) {
    initComp();
  }

  const rank = tensor.shape.length;

  const outputShape = new Array(rank);
  for (let i = 0; i < rank; i++) {
    outputShape[i] = tensor.shape[permutation[i]];
  }

  const inputStrides = computeStrides(tensor.shape);
  const mappedStrides = new Array(rank);
  for (let i = 0; i < rank; i++) {
    mappedStrides[i] = inputStrides[permutation[i]];
  }

  return compute(comp, outputShape, {
    x: tensor
  }, {
    mappedStrides: pad(mappedStrides)
  });
}