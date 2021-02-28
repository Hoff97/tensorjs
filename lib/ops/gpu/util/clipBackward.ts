import {defaultAllocator} from '../../../tensor/gpu/gl';
import {
  DTypeGpu,
  GPUTensorConstructor,
  GPUTensorI,
} from '../../../tensor/gpu/interface';
import {GPUMemoryAllocator} from '../../../tensor/gpu/memory';
import {Input, Operation} from '../operation';

export interface ClipBackwardInfo {
  shapeX?: readonly number[];
  widthX?: number;
  heightX?: number;

  shapeGrad?: readonly number[];
  widthGrad?: number;
  heightGrad?: number;

  shapeOutput?: readonly number[];
  widthOutput?: number;
  heightOutput?: number;

  minVal?: number;
  maxVal?: number;
  doMin?: number;
  doMax?: number;
}

export interface ClipBackwardInput {
  input: GPUTensorI;
  grad: GPUTensorI;
  minVal?: number;
  maxVal?: number;
}

export class ClipBackwardOperation<
  GPUTensor extends GPUTensorI
> extends Operation<GPUTensor, ClipBackwardInfo, ClipBackwardInput> {
  constructor(
    tensorConstructor: GPUTensorConstructor<GPUTensor>,
    dtype: DTypeGpu,
    allocator?: GPUMemoryAllocator
  ) {
    super(tensorConstructor, dtype, allocator);
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  getFragmentShader(info: ClipBackwardInfo): string {
    return `
    float process(int index[${this.maxRank}]) {
      float val = _X(index);
      if (doMin == 1 && val < minVal) {
        return 0.0;
      }
      if (doMax == 1 && val > maxVal) {
        return 0.0;
      }
      return _Grad(index);
    }

    ${this.getDefaultMain()}
    `;
  }

  getTextureNames(): string[] {
    return ['X', 'Grad'];
  }

  getVariables() {
    return `
    ${this.getVarModifier('minVal')} float minVal;
    ${this.getVarModifier('maxVal')} float maxVal;
    ${this.getVarModifier('doMin')} int doMin;
    ${this.getVarModifier('doMax')} int doMax;
    `;
  }

  getUniformAttrs(): Input[] {
    return [
      {name: 'minVal', type: 'float'},
      {name: 'maxVal', type: 'float'},
      {name: 'doMin'},
      {name: 'doMax'},
    ];
  }

  calc(input: ClipBackwardInput): GPUTensor {
    if (this.fullyStatic && this.outputShape !== undefined) {
      return this.compute(this.outputShape, {X: input.input, Grad: input.grad});
    }

    return this.compute(
      input.input.shape,
      {X: input.input, Grad: input.grad},
      {
        minVal: input.minVal !== undefined ? input.minVal : 0,
        maxVal: input.maxVal !== undefined ? input.maxVal : 0,
        doMin: input.minVal !== undefined ? 1 : 0,
        doMax: input.maxVal !== undefined ? 1 : 0,
      }
    );
  }

  getOutputShape(input: ClipBackwardInput): readonly number[] {
    return input.input.shape;
  }

  compile(info: ClipBackwardInfo) {
    if (info.shapeX !== undefined) {
      this.maxRank = info.shapeX.length;
    }

    super.compile(info);
  }

  getCompilationInfo(input: ClipBackwardInput): ClipBackwardInfo {
    const outputSize = defaultAllocator.getAllocationDimensions(
      input.input.size,
      this.dtype
    );

    return {
      shapeX: input.input.shape,
      widthX: input.input.memory.width,
      heightX: input.input.memory.height,

      shapeGrad: input.grad.shape,
      widthGrad: input.grad.memory.width,
      heightGrad: input.grad.memory.height,

      shapeOutput: input.input.shape,
      widthOutput: outputSize.width,
      heightOutput: outputSize.height,

      minVal: input.minVal !== undefined ? input.minVal : 0,
      maxVal: input.maxVal !== undefined ? input.maxVal : 0,
      doMin: input.minVal !== undefined ? 1 : 0,
      doMax: input.maxVal !== undefined ? 1 : 0,
    };
  }

  getInputInfoString(input: ClipBackwardInput): string {
    return `${input.input.shape}-${input.minVal}-${input.maxVal}`;
  }
}
