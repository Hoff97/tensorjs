import { DrawCommand } from "regl";
import GPUTensor from "../../tensor/gpu/tensor";
import { computeStrides } from "../../util/shape";
import { poolResultShape } from "../util/pool";
import { buildComp, compute, maxRank, defaultMain, initIndex, maxIterations, pad, posToIndex, incrementIndex, incrementConditional } from "./util";

let comp: DrawCommand;

const fragmentShader = `
uniform int mappedInputStrides[${maxRank}];
uniform int sumDims[${maxRank}];
uniform int sumSize;

float process(int index[${maxRank}]) {
  int inputIx[${maxRank}];
  ${initIndex('inputIx')}

  int inputPos = 0;
  for (int i = 0; i < ${maxRank}; i++) {
    if (mappedInputStrides[i] == -1 || index[i] == -1) {
      break;
    }
    inputPos += mappedInputStrides[i]*index[i];
  }

  ${posToIndex('stridesinput1', 'inputIx', 'inputPos')}

  float res = 0.0;

  for (int i = 0; i < ${maxIterations}; i++) {
    if (i >= sumSize) {
      break;
    }
    res += getValueAt(inputIx, stridesinput1, sizeinput1, input1);

    ${incrementConditional('inputIx', 'shapeinput1', 'sumDims')}
  }

  return res;
}

${defaultMain}
`;

function initComp() {
  comp = buildComp(['input1'], fragmentShader, [
    { name: 'mappedInputStrides', length: maxRank },
    { name: 'sumDims', length: maxRank },
    { name: 'sumSize' },
  ]);
}

export function sum(tensor1: GPUTensor, axes: number[]) {
  if (comp === undefined) {
    initComp();
  }

  const [outputShape, ixMap] = poolResultShape(tensor1.getShape(), axes);

  const inputStrides = computeStrides(tensor1.shape);
  const mappedInputStrides = [];
  for (let i of ixMap) {
    mappedInputStrides.push(inputStrides[i])
  }

  let sumSize = 1;
  const sumDims: number[] = new Array(tensor1.shape.length).fill(0);
  for (let i = 0; i < axes.length; i++) {
    sumDims[axes[i]] = 1;
    sumSize *= tensor1.shape[axes[i]];
  }

  return compute(comp, outputShape, {
    input1: tensor1,
  }, {
    mappedInputStrides: pad(mappedInputStrides),
    sumDims: pad(sumDims),
    sumSize
  });
}