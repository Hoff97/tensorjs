import {defaultAllocator} from '../../../tensor/gpu/gl';
import {
  DTypeGpu,
  GPUTensorConstructor,
  GPUTensorI,
} from '../../../tensor/gpu/interface';
import {GPUMemoryAllocator} from '../../../tensor/gpu/memory';
import {getSize} from '../../../util/shape';
import {outputDimsSize} from '../../util/convTranspose';
import {Input, Operation} from '../operation';

export interface ConvTransposeInfo {
  shapeX?: readonly number[];
  widthX?: number;
  heightX?: number;
  shapeW?: readonly number[];
  widthW?: number;
  heightW?: number;
  shapeOutput?: readonly number[];
  widthOutput?: number;
  heightOutput?: number;

  pads?: readonly number[];
  dilations?: readonly number[];
  strides?: readonly number[];

  CG?: number;
  kernelSize?: number;
  dataRank?: number;
  C?: number;
}

export interface ConvTransposeInput {
  X: GPUTensorI;
  W: GPUTensorI;
  pads: readonly number[];
  dilations: readonly number[];
  strides: readonly number[];
}

export class ConvTransposeOperation<
  GPUTensor extends GPUTensorI
> extends Operation<GPUTensor, ConvTransposeInfo, ConvTransposeInput> {
  protected maxIterations = 1000000;

  constructor(
    tensorConstructor: GPUTensorConstructor<GPUTensor>,
    dtype: DTypeGpu,
    allocator?: GPUMemoryAllocator
  ) {
    super(tensorConstructor, dtype, allocator);
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

      int trans_kernel_ix = shapeW[d + 2] - kernelIx[d + 2] - 1;

      inputIx[d+2] = index[d + 2] - pad + trans_kernel_ix * dilation;

      int divS = inputIx[d+2] / stride;
      int resS = inputIx[d+2] - divS*stride;

      if (resS != 0) {
        skip = true;
        break;
      }
      inputIx[d+2] = divS;

      if (inputIx[d+2] < 0 || inputIx[d+2] >= shapeX[d+2]) {
        skip = true;
        break;
      }
    }
    `;
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
    `;
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  getFragmentShader(info: ConvTransposeInfo): string {
    return `
    float process(int index[${this.maxRank}]) {
      float res = 0.0;

      ${this.getMainBody()}

      return res;
    }

    ${this.getDefaultMain()}
    `;
  }

  getTextureNames(): string[] {
    return ['X', 'W'];
  }

  getUniformAttrs(): Input[] {
    return [
      {name: 'CG'},
      {name: 'kernelSize'},
      {name: 'C'},
      {name: 'dataRank'},
      {name: 'pads', length: this.maxRank * 2},
      {name: 'strides', length: this.maxRank},
      {name: 'dilations', length: this.maxRank},
    ];
  }

  calc(input: ConvTransposeInput): GPUTensor {
    if (this.fullyStatic && this.outputShape !== undefined) {
      return this.compute(this.outputShape, {X: input.X, W: input.W});
    }

    const N = input.X.shape[0];
    const C = input.X.shape[1];
    const D = input.X.shape.slice(2);
    const W = input.W.shape.slice(2);
    const M = input.W.shape[0];
    const CG = input.W.shape[1];

    const kernelSize = getSize(W);

    const R = outputDimsSize(
      D,
      W,
      input.pads.slice(0, input.pads.length / 2),
      input.pads.slice(input.pads.length / 2),
      input.dilations,
      input.strides
    );
    let outputShape = [N, M];
    outputShape = outputShape.concat(R);

    return this.compute(
      outputShape,
      {X: input.X, W: input.W},
      {
        CG,
        kernelSize,
        C,
        dataRank: D.length,
        pads: this.copyPad(input.pads, this.maxRank * 2),
        strides: this.copyPad(input.strides),
        dilations: this.copyPad(input.dilations),
      }
    );
  }

  getOutputShape(input: ConvTransposeInput): readonly number[] {
    const N = input.X.shape[0];
    const D = input.X.shape.slice(2);
    const W = input.W.shape.slice(2);
    const M = input.W.shape[0];

    const R = outputDimsSize(
      D,
      W,
      input.pads.slice(0, input.pads.length / 2),
      input.pads.slice(input.pads.length / 2),
      input.dilations,
      input.strides
    );
    let outputShape = [N, M];
    outputShape = outputShape.concat(R);

    return outputShape;
  }

  compile(info: ConvTransposeInfo) {
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

    super.compile(info);
  }

  getCompilationInfo(input: ConvTransposeInput): ConvTransposeInfo {
    const outputShape = this.getOutputShape(input);
    const outputSize = defaultAllocator.getAllocationDimensions(
      getSize(outputShape),
      this.dtype
    );

    const kernelSize = getSize(input.W.shape.slice(2));

    const C = input.X.shape[1];
    const D = input.X.shape.slice(2);

    return {
      shapeX: input.X.shape,
      widthX: input.X.memory.width,
      heightX: input.X.memory.height,

      shapeW: input.W.shape,
      widthW: input.W.memory.width,
      heightW: input.W.memory.height,

      shapeOutput: outputShape,
      widthOutput: outputSize.width,
      heightOutput: outputSize.height,

      pads: input.pads,
      dilations: input.dilations,
      strides: input.strides,

      CG: input.W.shape[1],
      kernelSize: kernelSize,
      dataRank: D.length,
      C: C,
    };
  }

  getInputInfoString(input: ConvTransposeInput): string {
    return `${input.X.shape}-${input.W.shape}-${input.dilations}-${input.pads}-${input.dilations}-${input.strides}`;
  }
}
