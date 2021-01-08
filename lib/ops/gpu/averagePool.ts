import { DrawCommand } from "regl";
import { GPUTensor } from "../../tensor/gpu/tensor";
import { getSize } from "../../util/shape";
import { outputDimsSize } from "../util/conv";
import { buildComp, compute, maxRank, defaultMain, initIndex, maxIterations, incrementIndex, pad } from "./util";

let comp: DrawCommand;

const updateInputIx = `
for (int d = 0; d < ${maxRank - 2}; d++) {
  int stride = strides[d];
  int pad = pads[d];
  if (stride == -1) {
    break;
  }

  inputIx[d+2] = index[d+2]*stride - pad + kernelIx[d];
  if (inputIx[d+2] < 0 || inputIx[d+2] >= shapex[d+2]) {
    skip = true;
    break;
  }
}
`;

const variables = `
uniform int kernelSize;
uniform int dataRank;
uniform int includePad;
uniform int kernelShape[${maxRank}];
uniform int pads[${maxRank}];
uniform int strides[${maxRank}];
`;

const fragmentShader = `
${variables}

float process(int index[${maxRank}]) {
  float res = 0.0;
  int count = 0;

  int n = index[0];
  int c = index[1];

  int kernelIx[${maxRank}];
  ${initIndex('kernelIx')}
  for (int i = 0; i < ${maxRank}; i++) {
    if (i >= dataRank) {
      break;
    }
    kernelIx[i] = 0;
  }
  int inputIx[${maxRank}];
  ${initIndex('inputIx')}
  inputIx[0] = n;
  inputIx[1] = c;

  for (int kIx = 0; kIx < ${maxIterations}; kIx++) {
    if (kIx >= kernelSize) {
      break;
    }

    bool skip = false;

    ${updateInputIx}

    if (!skip) {
      res += _x(inputIx);
    }

    if (!skip || includePad == 1) {
      count += 1;
    }

    ${incrementIndex('kernelIx', 'kernelShape')}
  }

  res = res / float(count);

  return res;
}

${defaultMain}
`;

function initComp() {
  const args = [
    {name: 'kernelSize'},
    {name: 'dataRank'},
    {name: 'includePad'},
    {name: 'pads', length: maxRank},
    {name: 'strides', length: maxRank},
    {name: 'kernelShape', length: maxRank}
  ];

  comp = buildComp(['x'], fragmentShader, args);
}

export function averagePool(x: GPUTensor,
                            kernelShape: number[],
                            pads: number[],
                            strides: number[],
                            includePad: boolean) {
  if (comp === undefined) {
    initComp();
  }

  const dataRank = x.shape.length - 2;

  const N = x.shape[0];
  const C = x.shape[1];
  const D = x.shape.slice(2);

  const kernelSize = getSize(kernelShape);

  const R = outputDimsSize(D, kernelShape, pads.slice(0, pads.length/2), pads.slice(pads.length/2), new Array(dataRank).fill(1), strides);
  let outputShape = [N, C];
  outputShape = outputShape.concat(R);

  return compute(comp, outputShape, { x }, {
    kernelSize,
    kernelShape: pad(kernelShape),
    dataRank: D.length,
    pads: pad(pads.slice(0, pads.length/2)),
    strides: pad(strides),
    includePad: includePad ? 1 : 0
  });
}