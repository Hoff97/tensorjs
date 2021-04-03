import {defaultAllocator} from '../../../tensor/gpu/gl';
import {
  DTypeGpu,
  GPUTensorConstructor,
  GPUTensorI,
} from '../../../tensor/gpu/interface';
import {GPUMemoryAllocator} from '../../../tensor/gpu/memory';
import {gpuConstructor, GPUTensor} from '../../../tensor/gpu/tensor';
import {getSize} from '../../../util/shape';
import {Dispatcher} from '../../gpu/dispatcher';
import {Input, Operation} from '../../gpu/operation';

export interface AddIndexInfo {
  shapeA?: readonly number[];
  widthA?: number;
  heightA?: number;

  shapeOutput?: readonly number[];
  widthOutput?: number;
  heightOutput?: number;

  axis?: number;
  count?: number;
}

export interface AddIndexInput {
  A: GPUTensorI;
  axis: number;
  count: number;
}

export class AddIndexOperation<GPUTensor extends GPUTensorI> extends Operation<
  GPUTensor,
  AddIndexInfo,
  AddIndexInput
> {
  constructor(
    tensorConstructor: GPUTensorConstructor<GPUTensor>,
    dtype: DTypeGpu,
    allocator?: GPUMemoryAllocator
  ) {
    super(tensorConstructor, dtype, allocator);
  }

  getVariables() {
    return `
    ${this.getVarModifier('axis')} int axis;
    ${this.getVarModifier('count')} int count;
    `;
  }

  getUniformAttrs(): Input[] {
    return [{name: 'axis'}, {name: 'count'}];
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  getFragmentShader(info: AddIndexInfo): string {
    return `
    float process(int index[${this.maxRank}]) {
      float res = _A(index);

      if (index[1] == axis) {
        res += float(count);
      }

      return res;
    }

    ${this.getDefaultMain()}
    `;
  }

  getTextureNames(): string[] {
    return ['A'];
  }

  calc(input: AddIndexInput): GPUTensor {
    if (this.fullyStatic && this.outputShape !== undefined) {
      return this.compute(this.outputShape, {A: input.A});
    }

    const outputShape = this.getOutputShape(input);

    const info = this.getCompilationInfo(input);

    return this.compute(
      outputShape,
      {A: input.A},
      {
        axis: info.axis,
        count: info.count,
      }
    );
  }

  getOutputShape(input: AddIndexInput): readonly number[] {
    return [...input.A.shape];
  }

  compile(info: AddIndexInfo) {
    super.compile(info);
  }

  getCompilationInfo(input: AddIndexInput): AddIndexInfo {
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

      axis: input.axis,
      count: input.count,
    };
  }

  getInputInfoString(input: AddIndexInput): string {
    return `${input.A.shape}-${input.axis}-${input.count}`;
  }
}

export const defaultAddIndexD = new Dispatcher(
  (dtype: DTypeGpu) => new AddIndexOperation(gpuConstructor, dtype)
);

export function addIndexGPU(
  indices: GPUTensor<'uint32'>,
  axis: number,
  count: number
): GPUTensor<'uint32'> {
  return defaultAddIndexD.calc(
    {
      A: indices,
      axis,
      count,
    },
    'uint32'
  ) as GPUTensor<'uint32'>;
}
