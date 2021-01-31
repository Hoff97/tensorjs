import { getSize } from "../../../util/shape";
import { outputDimsSize } from "../../util/conv";

import { GPUTensorConstructor, GPUTensorI } from "../../../tensor/gpu/interface";
import { GPUMemoryAllocator } from "../../../tensor/gpu/memory";
import { Input, Operation } from "../operation";
import { Precision } from "../../../types";
import { defaultAllocator } from "../../../tensor/gpu/gl";


export interface AveragePoolInfo {
  shapeX?: readonly number[];
  widthX?: number;
  heightX?: number;

  kernelShape?: readonly number[];

  shapeOutput?: readonly number[],
  widthOutput?: number;
  heightOutput?: number;

  pads?: readonly number[];
  strides?: readonly number[];

  kernelSize?: number;
  dataRank?: number;
  includePad?: number | boolean;
}

export interface AveragePoolInput {
  X: GPUTensorI;
  pads: number[];
  strides: number[];
  kernelShape: number[];
  includePad: boolean;
}

export class AveragePoolOperation<GPUTensor extends GPUTensorI> extends Operation<GPUTensor, AveragePoolInfo, AveragePoolInput> {
  protected maxIterations = 1000000;

  constructor(tensorConstructor: GPUTensorConstructor<GPUTensor>, allocator?: GPUMemoryAllocator) {
    super(tensorConstructor, allocator);
  }

  updateInputIx() {
    return `
    for (int d = 0; d < ${this.maxRank - 2}; d++) {
      int stride = strides[d];
      int pad = pads[d];
      if (stride == -1) {
        break;
      }

      inputIx[d+2] = index[d+2]*stride - pad + kernelIx[d];
      if (inputIx[d+2] < 0 || inputIx[d+2] >= shapeX[d+2]) {
        skip = true;
        break;
      }
    }
    `
  }

  getVariables() {
    return `
    ${this.getVarModifier('kernelSize')} int kernelSize;
    ${this.getVarModifier('dataRank')} int dataRank;
    ${this.getVarModifier('includePad')} int includePad;
    ${this.getVarModifier('pads')} int pads[${this.maxRank}];
    ${this.getVarModifier('strides')} int strides[${this.maxRank}];
    ${this.getVarModifier('kernelShape')} int kernelShape[${this.maxRank}];
    `;
  }

  getFragmentShader(info: AveragePoolInfo): string {
    return `
    float process(int index[${this.maxRank}]) {
      float res = 0.0;
      int count = 0;

      int n = index[0];
      int c = index[1];

      int kernelIx[${this.maxRank}];
      ${this.initIndex('kernelIx')}
      for (int i = 0; i < ${this.maxRank}; i++) {
        if (i >= dataRank) {
          break;
        }
        kernelIx[i] = 0;
      }
      int inputIx[${this.maxRank}];
      ${this.initIndex('inputIx')}
      inputIx[0] = n;
      inputIx[1] = c;

      for (int kIx = 0; kIx < ${this.maxIterations}; kIx++) {
        if (kIx >= kernelSize) {
          break;
        }

        bool skip = false;

        ${this.updateInputIx()}

        if (!skip) {
          res += _X(inputIx);
        }

        if (!skip || includePad == 1) {
          count += 1;
        }

        ${this.incrementIndex('kernelIx', 'kernelShape')}
      }

      res = res / float(count);

      return res;
    }

    ${this.getDefaultMain()}
    `;
  }

  getTextureNames(): string[] {
    return ["X"];
  }

  getUniformAttrs(): Input[] {
    return [
      { name: "kernelSize" },
      { name: "dataRank" },
      { name: "includePad" },
      { name: "pads", length: this.maxRank*2 },
      { name: "strides", length: this.maxRank },
      { name: "kernelShape", length: this.maxRank }
    ];
  }

  calc(input: AveragePoolInput): GPUTensor {
    if (this.fullyStatic) {
      return this.compute(this.outputShape, {X: input.X});
    }
    const N = input.X.shape[0];
    const C = input.X.shape[1];
    const D = input.X.shape.slice(2);

    const kernelSize = getSize(input.kernelShape);

    const R = outputDimsSize(D, input.kernelShape, input.pads.slice(0, input.pads.length/2), input.pads.slice(input.pads.length/2), new Array(D.length).fill(1), input.strides);
    let outputShape = [N, C];
    outputShape = outputShape.concat(R);

    return this.compute(outputShape, {X: input.X}, {
      kernelSize, includePad: input.includePad ? 1 : 0,
      dataRank: D.length,
      pads: this.copyPad(input.pads, this.maxRank*2),
      strides: this.copyPad(input.strides),
      kernelShape: this.copyPad(input.kernelShape)
    })
  }

  getOutputShape(input: AveragePoolInput): readonly number[] {
    const N = input.X.shape[0];
    const C = input.X.shape[1];
    const D = input.X.shape.slice(2);

    const R = outputDimsSize(D, input.kernelShape, input.pads.slice(0, input.pads.length/2), input.pads.slice(input.pads.length/2), new Array(D.length).fill(1), input.strides);
    let outputShape = [N, C];
    outputShape = outputShape.concat(R);

    return outputShape;
  }

  compile(info: AveragePoolInfo, precision: Precision) {
    if (info.shapeX !== undefined) {
      info.dataRank = info.shapeX.length - 2;

      this.maxRank = info.shapeX.length;
    }
    if (info.includePad === true) {
      info.includePad = 1;
    } else if (info.includePad === false) {
      info.includePad = 0;
    }

    super.compile(info, precision);
  }

  getCompilationInfo(input: AveragePoolInput, precision: Precision): AveragePoolInfo {
    const outputShape = this.getOutputShape(input);
    const outputSize = defaultAllocator.getAllocationDimensions(getSize(outputShape), precision);

    const kernelSize = getSize(input.kernelShape);

    return {
      shapeX: input.X.shape,
      widthX: input.X.memory.width,
      heightX: input.X.memory.height,

      kernelShape: input.kernelShape,

      shapeOutput: outputShape,
      widthOutput: outputSize.width,
      heightOutput: outputSize.height,

      pads: input.pads,
      strides: input.strides,

      kernelSize: kernelSize,
      dataRank: input.X.shape.length - 2,
      includePad: input.includePad ? 1 : 0
    };
  }

  getInputInfoString(input: AveragePoolInput): string {
    return `${input.X.shape}-${input.kernelShape}-${input.pads}-${input.strides}-${input.includePad}`;
  }
}
