import { DrawCommand } from "regl";
import { GPUTensor } from "../../tensor/gpu/tensor";
import { PadMode } from "../../types";
import { buildComp, compute, copyPad, defaultMain, initIndex, maxRank } from "./util";

let compConst: DrawCommand;
let compReflect: DrawCommand;
let compEdge: DrawCommand;

const fragmentShaderConst = `
uniform int pads[${maxRank*2}];
uniform float value;

float process(int index[${maxRank}]) {
  float res = value;
  int inputIx[${maxRank}];
  ${initIndex('inputIx')}
  int outOfBounds = 0;
  for (int i = 0; i < ${maxRank}; i++) {
    if (index[i] == -1) {
      break;
    }
    inputIx[i] = index[i] - pads[i];
    if (inputIx[i] < 0 || inputIx[i] >= shapex[i]) {
      outOfBounds = 1;
      break;
    }
  }

  if (outOfBounds == 0) {
    res = _x(inputIx);
  }

  return res;
}

${defaultMain}
`;

const fragmentShaderReflect = `
uniform int pads[${maxRank*2}];

float process(int index[${maxRank}]) {
  int inputIx[${maxRank}];
  ${initIndex('inputIx')}
  for (int i = 0; i < ${maxRank}; i++) {
    if (index[i] == -1) {
      break;
    }
    inputIx[i] = index[i] - pads[i];
    if (inputIx[i] < 0) {
      inputIx[i] = -inputIx[i];
    } else if (inputIx[i] >= shapex[i]) {
      inputIx[i] = 2*shapex[i] - inputIx[i] - 2;
    }
  }

  return _x(inputIx);
}

${defaultMain}
`;

const fragmentShaderEdge = `
uniform int pads[${maxRank*2}];

float process(int index[${maxRank}]) {
  int inputIx[${maxRank}];
  ${initIndex('inputIx')}
  for (int i = 0; i < ${maxRank}; i++) {
    if (index[i] == -1) {
      break;
    }
    inputIx[i] = index[i] - pads[i];
    if (inputIx[i] < 0) {
      inputIx[i] = 0;
    } else if (inputIx[i] >= shapex[i]) {
      inputIx[i] = shapex[i] - 1;
    }
  }

  return _x(inputIx);
}

${defaultMain}
`;

function initComp() {
  compConst = buildComp(['x'], fragmentShaderConst, [{name: 'pads', length: 2*maxRank}, {name: 'value'}]);
  compReflect = buildComp(['x'], fragmentShaderReflect, [{name: 'pads', length: 2*maxRank}]);
  compEdge = buildComp(['x'], fragmentShaderEdge, [{name: 'pads', length: 2*maxRank}]);
}

export function padOp(tensor: GPUTensor, pads: number[], mode: PadMode, value: number) {
  if (compConst === undefined) {
    initComp();
  }

  const rank = tensor.shape.length;

  const resultShape = [...tensor.shape];
  for (let i = 0; i < rank; i++) {
    resultShape[i] += pads[i] + pads[i+rank];
  }

  if (mode === 'constant') {
    return compute(compConst, resultShape, {
      x: tensor,
    }, {
      pads: copyPad(pads, maxRank*2),
      value
    });
  } else if (mode === 'reflect') {
    return compute(compReflect, resultShape, {
      x: tensor,
    }, {
      pads: copyPad(pads, maxRank*2)
    });
  } else {
    return compute(compEdge, resultShape, {
      x: tensor,
    }, {
      pads: copyPad(pads, maxRank*2)
    });
  }
}