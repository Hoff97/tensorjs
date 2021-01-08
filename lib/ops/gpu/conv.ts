import { DrawCommand } from "regl";
import { GPUTensor } from "../../tensor/gpu/tensor";
import { getSize } from "../../util/shape";
import { outputDimsSize } from "../util/conv";
import { buildComp, compute, maxRank, defaultMain, initIndex, maxIterations, incrementIndex, pad } from "./util";

let comp: DrawCommand;
let compWithBias: DrawCommand;

const updateInputIx = `
for (int d = 0; d < ${maxRank - 2}; d++) {
  int stride = strides[d];
  int pad = pads[d];
  int dilation = dilations[d];
  if (stride == -1) {
    break;
  }

  inputIx[d+2] = index[d+2]*stride - pad + kernelIx[d+2]*dilation;
  if (inputIx[d+2] < 0 || inputIx[d+2] >= shapex[d+2]) {
    skip = true;
    break;
  }
}
`;

const variables = `
uniform int CG;
uniform int kernelSize;
uniform int dataRank;
uniform int C;
uniform int dilations[${maxRank}];
uniform int pads[${maxRank}];
uniform int strides[${maxRank}];
`;

const mainBody = `
  int n = index[0];
  int m = index[1];

  int kernelIx[${maxRank}];
  ${initIndex('kernelIx')}
  for (int i = 0; i < ${maxRank}; i++) {
    if (i >= dataRank) {
      break;
    }
    kernelIx[i+2] = 0;
  }
  kernelIx[0] = m;
  int inputIx[${maxRank}];
  ${initIndex('inputIx')}
  inputIx[0] = n;

  for (int cg = 0; cg < ${maxIterations}; cg++) {
    if (cg >= CG) {
      break;
    }
    int c = m * CG + cg;
    int d = c/C;
    c = c - d*C;
    inputIx[1] = c;
    kernelIx[1] = cg;
    for (int kIx = 0; kIx < ${maxIterations}; kIx++) {
      if (kIx >= kernelSize) {
        break;
      }

      bool skip = false;

      ${updateInputIx}

      if (!skip) {
        res += _x(inputIx) * _kernel(kernelIx);
      }

      ${incrementIndex('kernelIx', 'shapekernel')}
    }
  }
`;

const fragmentShaderBias = `
${variables}

float process(int index[${maxRank}]) {
  int biasIndex[${maxRank}];
  ${initIndex('biasIndex')}
  biasIndex[0] = index[1];
  float res = _bias(biasIndex);

  ${mainBody}

  return res;
}

${defaultMain}
`;

const fragmentShader = `
${variables}

float process(int index[${maxRank}]) {
  float res = 0.0;

  ${mainBody}

  return res;
}

${defaultMain}
`;

function initComp() {
  const args = [
    {name: 'CG'},
    {name: 'kernelSize'},
    {name: 'dataRank'},
    {name: 'C'},
    {name: 'dilations', length: maxRank},
    {name: 'pads', length: maxRank},
    {name: 'strides', length: maxRank}
  ];

  compWithBias = buildComp(['x', 'kernel', 'bias'], fragmentShaderBias, args);
  comp = buildComp(['x', 'kernel'], fragmentShader, args);
}

export function conv(x: GPUTensor,
                     kernel: GPUTensor,
                     dilations: number[],
                     group: number,
                     pads: number[],
                     strides: number[],
                     bias?: GPUTensor) {
  if (comp === undefined) {
    initComp();
  }

  const N = x.shape[0];
  const C = x.shape[1];
  const D = x.shape.slice(2);
  const W = kernel.shape.slice(2);
  const M = kernel.shape[0];
  const CG = kernel.shape[1];

  const kernelSize = getSize(W);

  const R = outputDimsSize(D, W, pads.slice(0, pads.length/2), pads.slice(pads.length/2), dilations, strides);
  let outputShape = [N, M];
  outputShape = outputShape.concat(R);

  const input: any = { x, kernel };
  if (bias) {
    input.bias = bias;
  }

  return compute(bias ? compWithBias : comp, outputShape, input, {
    kernelSize,
    CG: CG,
    dataRank: D.length,
    C: C,
    dilations: pad(dilations),
    pads: pad(pads.slice(0, pads.length/2)),
    strides: pad(strides)
  });
}