import { DrawCommand } from "regl";
import { GPUTensor } from "../../tensor/gpu/tensor";
import { buildComp, compute, copyPad, defaultMain, initIndex, maxRank } from "./util";

let comp: DrawCommand;

const fragmentShader = `
uniform int repeats[${maxRank}];

float process(int index[${maxRank}]) {
  int inIndex[${maxRank}];
  ${initIndex('inIndex')}
  for (int i = 0; i < ${maxRank}; i++) {
    if (repeats[i] == -1) {
      break;
    }
    int d = index[i] / shapex[i];
    inIndex[i] = index[i] - d*shapex[i];
  }

  return _x(inIndex);
}

${defaultMain}
`;

function initComp() {
  comp = buildComp(['x'], fragmentShader, [{name: 'repeats', length: maxRank}]);
}

export function repeat(tensor: GPUTensor, repeats: number[]) {
  if (comp === undefined) {
    initComp();
  }

  const rank = tensor.shape.length;

  const outputShape = new Array(rank);
  for (let i = 0; i < rank; i++) {
    outputShape[i] = tensor.shape[i] * repeats[i];
  }

  return compute(comp, outputShape, {
    x: tensor
  }, {
    repeats: copyPad(repeats)
  });
}