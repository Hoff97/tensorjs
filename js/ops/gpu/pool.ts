import { DrawCommand } from "regl";
import GPUTensor from "../../tensor/gpu/tensor";
import { computeStrides } from "../../util/shape";
import { poolResultShape } from "../util/pool";
import { buildComp, compute, maxRank, defaultMain, initIndex, maxIterations, pad, posToIndex, incrementIndex, incrementConditional } from "./util";


export function fragmentShader(update: (a: string, b: string) => string) {
  return `
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
    float curr = getValueAt(inputIx, stridesinput1, sizeinput1, input1);
    if (i == 0) {
      res = curr;
    } else {
      res = ${update('res', 'curr')};
    }

    ${incrementConditional('inputIx', 'shapeinput1', 'sumDims')}
  }

  return res;
}

${defaultMain}
`;
}

export function initComputation(fragShader: string) {
  return buildComp(['input1'], fragShader, [
    { name: 'mappedInputStrides', length: maxRank },
    { name: 'sumDims', length: maxRank },
    { name: 'sumSize' },
  ]);
}

export function performComputation(tensor1: GPUTensor,
                                   axes: number[],
                                   comp: any) {
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