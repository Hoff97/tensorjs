import {defaultAllocator} from '../../../tensor/gpu/gl';
import {
  DTypeGpu,
  GPUTensorConstructor,
  GPUTensorI,
} from '../../../tensor/gpu/interface';
import {GPUMemoryAllocator} from '../../../tensor/gpu/memory';
import {getSize} from '../../../util/shape';
import {Operation} from '../operation';

export interface CopyInfo {
  shapeX?: readonly number[];
  widthX?: number;
  heightX?: number;
  shapeOutput?: readonly number[];
  widthOutput?: number;
  heightOutput?: number;
}

export interface CopyInput {
  input: GPUTensorI;
  outputShape?: readonly number[];
}

export class CopyOperation<GPUTensor extends GPUTensorI> extends Operation<
  GPUTensor,
  CopyInfo,
  CopyInput
> {
  constructor(
    tensorConstructor: GPUTensorConstructor<GPUTensor>,
    dtype: DTypeGpu,
    allocator?: GPUMemoryAllocator
  ) {
    super(tensorConstructor, dtype, allocator);
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  getFragmentShader(info: CopyInfo): string {
    return `
    void main() {
      initVars();

      gl_FragColor = texture2D(X, uv);
    }
    `;
  }

  getTextureNames(): string[] {
    return ['X'];
  }

  calc(input: CopyInput): GPUTensor {
    const shape = this.getOutputShape(input);

    return this.compute(shape, {X: input.input});
  }

  getOutputShape(input: CopyInput): readonly number[] {
    let shape = input.outputShape;
    if (shape === undefined) {
      shape = input.input.shape;
    }
    return shape;
  }

  compile(info: CopyInfo) {
    if (info.shapeX !== undefined) {
      this.maxRank = info.shapeX.length;
    }

    super.compile(info);
  }

  getCompilationInfo(input: CopyInput): CopyInfo {
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

  getInputInfoString(input: CopyInput): string {
    return `${input.input}-${this.getOutputShape(input)}`;
  }
}
