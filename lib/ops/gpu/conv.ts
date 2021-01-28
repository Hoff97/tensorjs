import { getSize } from "../../util/shape";
import { outputDimsSize } from "../util/conv";

import { GPUTensorConstructor, GPUTensorI } from "../../tensor/gpu/interface";
import { GPUMemoryAllocator } from "../../tensor/gpu/memory";
import { Input, Operation } from "./operation";
import { Activation, Precision } from "../../types";


export interface ConvInfo {
  shapeX?: readonly number[];
  widthX?: number;
  heightX?: number;
  shapeW?: readonly number[];
  widthW?: number;
  heightW?: number;
  shapeOutput?: readonly number[],
  widthOutput?: number;
  heightOutput?: number;

  pads?: readonly number[];
  dilations?: readonly number[];
  strides?: readonly number[];

  CG?: number;
  kernelSize?: number;
  dataRank?: number;
  C?: number;
  activation?: Activation | number;
}

export interface ConvInput {
  X: GPUTensorI;
  W: GPUTensorI;
  pads: readonly number[];
  dilations: readonly number[];
  strides: readonly number[];
  activation: Activation;
}

export class ConvOperation<GPUTensor extends GPUTensorI, ConvInf extends ConvInfo = ConvInfo, ConvIn extends ConvInput = ConvInput> extends Operation<GPUTensor, ConvInf, ConvIn> {
  protected maxIterations = 1000000;

  constructor(tensorConstructor: GPUTensorConstructor<GPUTensor>, allocator?: GPUMemoryAllocator) {
    super(tensorConstructor, allocator);
  }

  updateInputIx() {
    return `
    for (int d = 0; d < ${this.maxRank - 2}; d++) {
      int stride = strides[d];
      int pad = pads[d];
      int dilation = dilations[d];
      if (stride == -1) {
        break;
      }

      inputIx[d+2] = index[d+2]*stride - pad + kernelIx[d+2]*dilation;
      if (inputIx[d+2] < 0 || inputIx[d+2] >= shapeX[d+2]) {
        skip = true;
        break;
      }
    }
    `
  }

  getMainBody() {
    return `
    int n = index[0];
    int m = index[1];

    int kernelIx[${this.maxRank}];
    ${this.initIndex('kernelIx')}
    for (int i = 0; i < ${this.maxRank}; i++) {
      if (i >= dataRank) {
        break;
      }
      kernelIx[i+2] = 0;
    }
    kernelIx[0] = m;
    int inputIx[${this.maxRank}];
    ${this.initIndex('inputIx')}
    inputIx[0] = n;

    for (int cg = 0; cg < ${this.maxIterations}; cg++) {
      if (cg >= CG) {
        break;
      }
      int c = m * CG + cg;
      int d = c/C;
      c = c - d*C;
      inputIx[1] = c;
      kernelIx[1] = cg;
      for (int kIx = 0; kIx < ${this.maxIterations}; kIx++) {
        if (kIx >= kernelSize) {
          break;
        }

        bool skip = false;

        ${this.updateInputIx()}

        if (!skip) {
          res += _X(inputIx) * _W(kernelIx);
        }

        ${this.incrementIndex('kernelIx', 'shapeW')}
      }
    }
    `;
  }

  getVariables() {
    return `
    ${this.getVarModifier('CG')} int CG;
    ${this.getVarModifier('kernelSize')} int kernelSize;
    ${this.getVarModifier('dataRank')} int dataRank;
    ${this.getVarModifier('C')} int C;
    ${this.getVarModifier('dilations')} int dilations[${this.maxRank}];
    ${this.getVarModifier('pads')} int pads[${this.maxRank}];
    ${this.getVarModifier('strides')} int strides[${this.maxRank}];
    ${this.getVarModifier('activation')} int activation;
    `;
  }

  getFragmentShader(info: ConvInfo): string {
    return `
    float process(int index[${this.maxRank}]) {
      float res = 0.0;

      ${this.getMainBody()}

      if (activation == 1) {
        res = max(0.0, res);
      }

      return res;
    }

    ${this.getDefaultMain()}
    `;
  }

  getTextureNames(): string[] {
    return ["X", "W"];
  }

  getUniformAttrs(): Input[] {
    return [
      { name: "CG" },
      { name: "kernelSize" },
      { name: "C" },
      { name: "dataRank" },
      { name: "pads", length: this.maxRank*2 },
      { name: "strides", length: this.maxRank },
      { name: "dilations", length: this.maxRank },
      { name: "activation" }
    ];
  }

  getActivationFlag(activation: Activation) {
    if (activation === "id") {
      return 0;
    } else if (activation === "relu") {
      return 1;
    }
  }

  calc(input: ConvInput): GPUTensor {
    const N = input.X.shape[0];
    const C = input.X.shape[1];
    const D = input.X.shape.slice(2);
    const W = input.W.shape.slice(2);
    const M = input.W.shape[0];
    const CG = input.W.shape[1];

    const kernelSize = getSize(W);

    const R = outputDimsSize(D, W, input.pads.slice(0, input.pads.length/2), input.pads.slice(input.pads.length/2), input.dilations, input.strides);
    let outputShape = [N, M];
    outputShape = outputShape.concat(R);

    return this.compute(outputShape, {X: input.X, W: input.W}, {
      CG, kernelSize, C,
      dataRank: D.length,
      pads: this.copyPad(input.pads, this.maxRank*2),
      strides: this.copyPad(input.strides),
      dilations: this.copyPad(input.dilations),
      activation: this.getActivationFlag(input.activation)
    })
  }

  getOutputShape(input: ConvIn): readonly number[] {
    const N = input.X.shape[0];
    const D = input.X.shape.slice(2);
    const W = input.W.shape.slice(2);
    const M = input.W.shape[0];

    const R = outputDimsSize(D, W, input.pads.slice(0, input.pads.length/2), input.pads.slice(input.pads.length/2), input.dilations, input.strides);
    let outputShape = [N, M];
    outputShape = outputShape.concat(R);

    return outputShape;
  }

  compile(info: ConvInf, precision: Precision) {
    if (info.shapeW !== undefined) {
      info.CG = info.shapeW[1];
      info.kernelSize = getSize(info.shapeW.slice(2));
      info.dataRank = info.shapeW.length - 2;
      this.maxRank = info.shapeW.length;
    }
    if (info.shapeX !== undefined) {
      info.C = info.shapeX[1];
      info.dataRank = info.shapeX.length - 2;

      this.maxRank = info.shapeX.length;
    }
    if (info.activation !== undefined && typeof info.activation === "string") {
      info.activation = this.getActivationFlag(info.activation);
    }

    super.compile(info, precision);
  }
}


export interface ConvBiasInput extends ConvInput {
  B: GPUTensorI;
}

export interface ConvBiasInfo extends ConvInfo {
  shapeB?: number[];
  widthB?: number;
  heightB?: number;
}

export class ConvBiasOperation<GPUTensor extends GPUTensorI> extends ConvOperation<GPUTensor, ConvBiasInfo, ConvBiasInput> {
  protected maxIterations = 1000000;

  constructor(tensorConstructor: GPUTensorConstructor<GPUTensor>, allocator?: GPUMemoryAllocator) {
    super(tensorConstructor, allocator);
  }

  getFragmentShader(info: ConvBiasInfo): string {
    return `
    float process(int index[${this.maxRank}]) {
      int biasIndex[${this.maxRank}];
      ${this.initIndex('biasIndex')}
      biasIndex[0] = index[1];
      float res = _B(biasIndex);

      ${this.getMainBody()}

      if (activation == 1) {
        res = max(0.0, res);
      }

      return res;
    }

    ${this.getDefaultMain()}
    `;
  }

  getTextureNames(): string[] {
    return ["X", "W", "B"];
  }

  calc(input: ConvBiasInput): GPUTensor {
    const N = input.X.shape[0];
    const C = input.X.shape[1];
    const D = input.X.shape.slice(2);
    const W = input.W.shape.slice(2);
    const M = input.W.shape[0];
    const CG = input.W.shape[1];

    const kernelSize = getSize(W);

    const R = outputDimsSize(D, W, input.pads.slice(0, input.pads.length/2), input.pads.slice(input.pads.length/2), input.dilations, input.strides);
    let outputShape = [N, M];
    outputShape = outputShape.concat(R);

    return this.compute(outputShape, {X: input.X, W: input.W, B: input.B}, {
      CG, kernelSize, C,
      dataRank: D.length,
      pads: this.copyPad(input.pads, this.maxRank*2),
      strides: this.copyPad(input.strides),
      dilations: this.copyPad(input.dilations),
      activation: this.getActivationFlag(input.activation)
    })
  }
}
