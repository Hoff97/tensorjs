import {defaultAllocator} from '../../../tensor/gpu/gl';
import {
  DTypeGpu,
  GPUTensorConstructor,
  GPUTensorI,
} from '../../../tensor/gpu/interface';
import {GPUMemoryAllocator} from '../../../tensor/gpu/memory';
import {getSize} from '../../../util/shape';
import {Operation} from '../operation';

export interface ExpandInfo {
  shapeX?: readonly number[];
  widthX?: number;
  heightX?: number;
  shapeOutput?: readonly number[];
  widthOutput?: number;
  heightOutput?: number;
}

export interface ExpandInput {
  input: GPUTensorI;
  outputShape: readonly number[];
}

export class ExpandOperation<GPUTensor extends GPUTensorI> extends Operation<
  GPUTensor,
  ExpandInfo,
  ExpandInput
> {
  constructor(
    tensorConstructor: GPUTensorConstructor<GPUTensor>,
    dtype: DTypeGpu,
    allocator?: GPUMemoryAllocator
  ) {
    super(tensorConstructor, dtype, allocator);
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  getFragmentShader(info: ExpandInfo): string {
    return `
    float process(int index[${this.maxRank}]) {
      return _X(index);
    }

    ${this.getDefaultMain()}
    `;
  }

  getTextureNames(): string[] {
    return ['X'];
  }

  calc(input: ExpandInput): GPUTensor {
    return this.compute(input.outputShape, {X: input.input});
  }

  getOutputShape(input: ExpandInput): readonly number[] {
    return input.outputShape;
  }

  compile(info: ExpandInfo) {
    if (info.shapeX !== undefined) {
      this.maxRank = info.shapeX.length;
    }

    super.compile(info);
  }

  getCompilationInfo(input: ExpandInput): ExpandInfo {
    const outputShape = this.getOutputShape(input);
    const outputSize = defaultAllocator.getAllocationDimensions(
      getSize(outputShape),
      this.dtype
    );

    return {
      shapeX: input.input.shape,
      widthX: input.input.memory.width,
      heightX: input.input.memory.height,

      shapeOutput: outputShape,
      widthOutput: outputSize.width,
      heightOutput: outputSize.height,
    };
  }

  getInputInfoString(input: ExpandInput): string {
    return `${input.input.shape}-${input.outputShape}`;
  }
}
