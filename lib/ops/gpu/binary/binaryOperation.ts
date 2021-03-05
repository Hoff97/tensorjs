import {defaultAllocator} from '../../../tensor/gpu/gl';
import {
  DTypeGpu,
  GPUTensorConstructor,
  GPUTensorI,
} from '../../../tensor/gpu/interface';
import {GPUMemoryAllocator} from '../../../tensor/gpu/memory';
import {getSize} from '../../../util/shape';
import {Operation} from './../operation';

export interface BinaryOpInfo {
  shapeA?: readonly number[];
  widthA?: number;
  heightA?: number;
  shapeB?: readonly number[];
  widthB?: number;
  heightB?: number;
  shapeOutput?: readonly number[];
  widthOutput?: number;
  heightOutput?: number;
}

export interface BinaryOpInput {
  A: GPUTensorI;
  B: GPUTensorI;
  outputShape: readonly number[];
}

export abstract class BinaryOperation<
  GPUTensor extends GPUTensorI,
  BInfo extends BinaryOpInfo = BinaryOpInfo,
  BInput extends BinaryOpInput = BinaryOpInput
> extends Operation<GPUTensor, BInfo, BInput> {
  constructor(
    tensorConstructor: GPUTensorConstructor<GPUTensor>,
    dtype: DTypeGpu,
    allocator?: GPUMemoryAllocator
  ) {
    super(tensorConstructor, dtype, allocator);
  }

  abstract getOp(a: string, b: string): string;

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  getFragmentShader(info: BInfo): string {
    return `
    float process(int[${this.maxRank}] index) {
      return ${this.getOp('_A(index)', '_B(index)')};
    }

    ${this.getDefaultMain()}
    `;
  }

  getOutputShape(input: BInput): readonly number[] {
    return input.outputShape;
  }

  getTextureNames(): string[] {
    return ['A', 'B'];
  }

  calc(input: BInput): GPUTensor {
    return this.compute(input.outputShape, {A: input.A, B: input.B});
  }

  compile(info: BInfo) {
    if (info.shapeA !== undefined) {
      this.maxRank = info.shapeA.length;
    }
    if (info.shapeB !== undefined) {
      this.maxRank = info.shapeB.length;
    }

    super.compile(info);
  }

  getCompilationInfo(input: BInput): BInfo {
    const outputSize = defaultAllocator.getAllocationDimensions(
      getSize(input.outputShape),
      this.dtype
    );

    return {
      shapeA: input.A.shape,
      widthA: input.A.memory.width,
      heightA: input.A.memory.height,
      shapeB: input.B.shape,
      widthB: input.B.memory.width,
      heightB: input.B.memory.height,
      shapeOutput: input.outputShape,
      widthOutput: outputSize.width,
      heightOutput: outputSize.height,
    } as BInfo;
  }

  getInputInfoString(input: BinaryOpInput): string {
    return `${input.A.shape}-${input.B.shape}`;
  }
}
