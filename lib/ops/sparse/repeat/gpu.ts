import {defaultAllocator} from '../../../tensor/gpu/gl';
import {
  DTypeGpu,
  GPUTensorConstructor,
  GPUTensorI,
} from '../../../tensor/gpu/interface';
import {GPUMemoryAllocator} from '../../../tensor/gpu/memory';
import {gpuConstructor, GPUTensor} from '../../../tensor/gpu/tensor';
import {computeStrides, getSize} from '../../../util/shape';
import {Dispatcher} from '../../gpu/dispatcher';
import {Input, Operation} from '../../gpu/operation';

export interface RepeatIndexInfo {
  shapeA?: readonly number[];
  widthA?: number;
  heightA?: number;

  shapeOutput?: readonly number[];
  widthOutput?: number;
  heightOutput?: number;

  sparseShape?: readonly number[];
  repeatStrides?: readonly number[];
}

export interface RepeatIndexInput {
  A: GPUTensorI;
  repeats: readonly number[];
  shape: readonly number[];
  repeatsProd: number;
}

export class RepeatIndexOperation<
  GPUTensor extends GPUTensorI
> extends Operation<GPUTensor, RepeatIndexInfo, RepeatIndexInput> {
  constructor(
    tensorConstructor: GPUTensorConstructor<GPUTensor>,
    dtype: DTypeGpu,
    allocator?: GPUMemoryAllocator
  ) {
    super(tensorConstructor, dtype, allocator);
  }

  getVariables() {
    return `
    ${this.getVarModifier('sparseShape')} int sparseShape[${this.maxRank}];
    ${this.getVarModifier('repeatStrides')} int repeatStrides[${this.maxRank}];
    `;
  }

  getUniformAttrs(): Input[] {
    return [
      {name: 'sparseShape', length: this.maxRank},
      {name: 'repeatStrides', length: this.maxRank},
    ];
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  getFragmentShader(info: RepeatIndexInfo): string {
    return `
    float process(int index[${this.maxRank}]) {
      int newPos = index[0];
      int nnz = shapeA[0];

      int repeatPos = newPos / nnz;
      int oldPos = newPos - repeatPos*nnz;

      int oldIx[${this.maxRank}];
      ${this.initIndex('oldIx')}
      oldIx[0] = oldPos;
      oldIx[1] = index[1];

      int repeatIx[${this.maxRank}];
      ${this.initIndex('repeatIx')}
      ${this.posToIndex('repeatStrides', 'repeatIx', 'repeatPos')}

      float res = _A(oldIx);
      for (int i = 0; i < ${this.maxRank}; i++) {
        if (i == index[1]) {
          res += float(repeatIx[i]*sparseShape[i]);
          break;
        }
      }

      return res;
    }

    ${this.getDefaultMain()}
    `;
  }

  getTextureNames(): string[] {
    return ['A'];
  }

  calc(input: RepeatIndexInput): GPUTensor {
    if (this.fullyStatic && this.outputShape !== undefined) {
      return this.compute(this.outputShape, {A: input.A});
    }

    const outputShape = this.getOutputShape(input);

    const info = this.getCompilationInfo(input);

    return this.compute(
      outputShape,
      {A: input.A},
      {
        sparseShape: this.pad(info.sparseShape as number[]),
        repeatStrides: this.pad(info.repeatStrides as number[]),
      }
    );
  }

  getOutputShape(input: RepeatIndexInput): readonly number[] {
    return [input.A.shape[0] * input.repeatsProd, input.A.shape[1]];
  }

  compile(info: RepeatIndexInfo) {
    if (info.sparseShape !== undefined) {
      this.maxRank = info.sparseShape.length;
    }
    if (info.repeatStrides !== undefined) {
      this.maxRank = info.repeatStrides.length;
    }

    super.compile(info);
  }

  getCompilationInfo(input: RepeatIndexInput): RepeatIndexInfo {
    const outputShape = this.getOutputShape(input);
    const outputSize = defaultAllocator.getAllocationDimensions(
      getSize(outputShape),
      this.dtype
    );

    return {
      shapeA: input.A.shape,
      widthA: input.A.memory.width,
      heightA: input.A.memory.height,

      shapeOutput: outputShape,
      widthOutput: outputSize.width,
      heightOutput: outputSize.height,

      sparseShape: input.shape,
      repeatStrides: computeStrides(input.repeats),
    };
  }

  getInputInfoString(input: RepeatIndexInput): string {
    return `${input.A.shape}-${input.repeats}-${input.shape}`;
  }
}

export const defaultRepeatIndexD = new Dispatcher(
  (dtype: DTypeGpu) => new RepeatIndexOperation(gpuConstructor, dtype)
);

export function repeatIndexGPU(
  indices: GPUTensor<'uint32'>,
  repeats: readonly number[],
  shape: readonly number[],
  repeatsProd: number
): GPUTensor<'uint32'> {
  return defaultRepeatIndexD.calc(
    {
      A: indices,
      repeats,
      shape,
      repeatsProd,
    },
    'uint32'
  ) as GPUTensor<'uint32'>;
}
