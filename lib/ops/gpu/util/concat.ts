import {defaultAllocator} from '../../../tensor/gpu/gl';
import {GPUTensorConstructor, GPUTensorI} from '../../../tensor/gpu/interface';
import {GPUMemoryAllocator} from '../../../tensor/gpu/memory';
import {Precision} from '../../../types';
import {getSize} from '../../../util/shape';
import {Input, Operation} from '../operation';

export interface ConcatInfo {
  shapeA?: readonly number[];
  widthA?: number;
  heightA?: number;

  shapeB?: readonly number[];
  widthB?: number;
  heightB?: number;

  shapeOutput?: readonly number[];
  widthOutput?: number;
  heightOutput?: number;

  axis?: number;
}

export interface ConcatInput {
  A: GPUTensorI;
  B: GPUTensorI;
  axis: number;
}

export class ConcatOperation<GPUTensor extends GPUTensorI> extends Operation<
  GPUTensor,
  ConcatInfo,
  ConcatInput
> {
  constructor(
    tensorConstructor: GPUTensorConstructor<GPUTensor>,
    allocator?: GPUMemoryAllocator
  ) {
    super(tensorConstructor, allocator);
  }

  getVariables() {
    return `
    ${this.getVarModifier('axis')} int axis;
    `;
  }

  getUniformAttrs(): Input[] {
    return [{name: 'axis'}];
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  getFragmentShader(info: ConcatInfo): string {
    return `
    float process(int index[${this.maxRank}]) {
      float res = 0.0;
      for (int i = 0; i < ${this.maxRank}; i++) {
        if (i == axis) {
          if (index[i] >= shapeA[i]) {
            index[i] = index[i] - shapeA[i];
            res = _B(index);
          } else {
            res = _A(index);
          }
          break;
        }
      }
      return res;
    }

    ${this.getDefaultMain()}
    `;
  }

  getTextureNames(): string[] {
    return ['A', 'B'];
  }

  calc(input: ConcatInput): GPUTensor {
    if (this.fullyStatic && this.outputShape !== undefined) {
      return this.compute(this.outputShape, {A: input.A, B: input.B});
    }

    const outputShape = this.getOutputShape(input);

    return this.compute(
      outputShape,
      {A: input.A, B: input.B},
      {axis: input.axis}
    );
  }

  getOutputShape(input: ConcatInput): readonly number[] {
    const outputShape = [...input.A.shape];
    outputShape[input.axis] += input.B.shape[input.axis];
    return outputShape;
  }

  compile(info: ConcatInfo, precision: Precision) {
    if (info.shapeA !== undefined) {
      this.maxRank = info.shapeA.length;
    }
    if (info.shapeB !== undefined) {
      this.maxRank = info.shapeB.length;
    }

    super.compile(info, precision);
  }

  getCompilationInfo(input: ConcatInput, precision: Precision): ConcatInfo {
    const outputShape = this.getOutputShape(input);
    const outputSize = defaultAllocator.getAllocationDimensions(
      getSize(outputShape),
      precision
    );

    return {
      shapeA: input.A.shape,
      widthA: input.A.memory.width,
      heightA: input.A.memory.height,

      shapeB: input.B.shape,
      widthB: input.B.memory.width,
      heightB: input.B.memory.height,

      shapeOutput: outputShape,
      widthOutput: outputSize.width,
      heightOutput: outputSize.height,

      axis: input.axis,
    };
  }

  getInputInfoString(input: ConcatInput): string {
    return `${input.A.shape}-${input.B.shape}-${input.axis}`;
  }
}
