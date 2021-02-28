import {defaultAllocator} from '../../../tensor/gpu/gl';
import {
  DTypeGpu,
  GPUTensorConstructor,
  GPUTensorI,
} from '../../../tensor/gpu/interface';
import {GPUMemoryAllocator} from '../../../tensor/gpu/memory';
import {getSize} from '../../../util/shape';
import {Operation} from '../operation';

export interface MatMulInfo {
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

export interface MatMulInput {
  A: GPUTensorI;
  B: GPUTensorI;
}

export class MatMulOperation<GPUTensor extends GPUTensorI> extends Operation<
  GPUTensor,
  MatMulInfo,
  MatMulInput
> {
  protected maxIterations = 1000000;

  constructor(
    tensorConstructor: GPUTensorConstructor<GPUTensor>,
    dtype: DTypeGpu,
    allocator?: GPUMemoryAllocator
  ) {
    super(tensorConstructor, dtype, allocator);

    this.maxRank = 2;
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  getFragmentShader(info: MatMulInfo): string {
    return `
    float process(int index[${this.maxRank}]) {
      int ix1[${this.maxRank}];
      ${this.initIndex('ix1')}
      ix1[0] = index[0];

      int ix2[${this.maxRank}];
      ${this.initIndex('ix2')}
      ix2[1] = index[1];

      int k = shapeA[1];

      float res = 0.0;

      for (int i = 0; i < ${this.maxIterations}; i++) {
        if (i >= k) {
          break;
        }
        ix1[1] = i;
        ix2[0] = i;

        float v1 = _A(ix1);
        float v2 = _B(ix2);
        res += v1*v2;
      }

      return res;
    }

    ${this.getDefaultMain()}
    `;
  }

  getTextureNames(): string[] {
    return ['A', 'B'];
  }

  calc(input: MatMulInput): GPUTensor {
    const outputShape = this.getOutputShape(input);

    return this.compute(outputShape, {A: input.A, B: input.B});
  }

  getOutputShape(input: MatMulInput): readonly number[] {
    return [input.A.shape[0], input.B.shape[1]];
  }

  compile(info: MatMulInfo) {
    if (info.shapeA !== undefined) {
      this.maxIterations = info.shapeA[1];
    } else if (info.shapeB !== undefined) {
      this.maxIterations = info.shapeB[0];
    }

    super.compile(info);
  }

  getCompilationInfo(input: MatMulInput): MatMulInfo {
    const outputShape = this.getOutputShape(input);
    const outputSize = defaultAllocator.getAllocationDimensions(
      getSize(outputShape),
      this.dtype
    );

    return {
      shapeA: input.A.shape,
      widthA: input.A.memory.width,
      heightA: input.A.memory.height,

      shapeB: input.A.shape,
      widthB: input.A.memory.width,
      heightB: input.A.memory.height,

      shapeOutput: outputShape,
      widthOutput: outputSize.width,
      heightOutput: outputSize.height,
    };
  }

  getInputInfoString(input: MatMulInput): string {
    return `${input.A.shape}-${input.B.shape}`;
  }
}
