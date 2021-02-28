import {defaultAllocator} from '../../../tensor/gpu/gl';
import {
  DTypeGpu,
  GPUTensorConstructor,
  GPUTensorI,
} from '../../../tensor/gpu/interface';
import {GPUMemoryAllocator} from '../../../tensor/gpu/memory';
import {getSize} from '../../../util/shape';
import {Operation} from '../operation';

export interface UnaryOpInfo {
  shapeX?: readonly number[];
  widthX?: number;
  heightX?: number;
  shapeOutput?: readonly number[];
  widthOutput?: number;
  heightOutput?: number;
}

export interface UnaryOpInput {
  input: GPUTensorI;
}

export abstract class UnaryOperation<
  GPUTensor extends GPUTensorI,
  UInfo extends UnaryOpInfo = UnaryOpInfo,
  UInput extends UnaryOpInput = UnaryOpInput
> extends Operation<GPUTensor, UInfo, UInput> {
  constructor(
    tensorConstructor: GPUTensorConstructor<GPUTensor>,
    dtype: DTypeGpu,
    allocator?: GPUMemoryAllocator
  ) {
    super(tensorConstructor, dtype, allocator);
  }

  abstract operation(input: string): string;

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  getFragmentShader(info: UInfo): string {
    return `
    void main() {
      initVars();

      gl_FragColor = ${this.operation('texture2D(X, uv)')};
    }
    `;
  }

  getTextureNames(): string[] {
    return ['X'];
  }

  calc(input: UInput): GPUTensor {
    return this.compute(input.input.shape, {X: input.input});
  }

  getOutputShape(input: UInput): readonly number[] {
    return input.input.shape;
  }

  compile(info: UInfo) {
    if (info.shapeX !== undefined) {
      this.maxRank = info.shapeX.length;
    }

    super.compile(info);
  }

  getCompilationInfo(input: UInput): UInfo {
    const outputShape = this.getOutputShape(input);
    const outputSize = defaultAllocator.getAllocationDimensions(
      getSize(outputShape),
      this.dtype
    );

    return {
      shapeX: input.input.shape,
      widthX: input.input.memory.width,
      heightX: input.input.memory.height,
      shapeOutput: this.getOutputShape(input),
      widthOutput: outputSize.width,
      heightOutput: outputSize.height,
    } as UInfo;
  }

  getInputInfoString(input: UInput): string {
    return `${input.input.shape}`;
  }
}
