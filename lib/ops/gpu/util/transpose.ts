import {defaultAllocator} from '../../../tensor/gpu/gl';
import {
  DTypeGpu,
  GPUTensorConstructor,
  GPUTensorI,
} from '../../../tensor/gpu/interface';
import {GPUMemoryAllocator} from '../../../tensor/gpu/memory';
import {computeStrides, getSize} from '../../../util/shape';
import {Input, Operation} from '../operation';

export interface TransposeInfo {
  shapeA?: readonly number[];
  widthA?: number;
  heightA?: number;

  shapeOutput?: readonly number[];
  widthOutput?: number;
  heightOutput?: number;

  permutation?: readonly number[];
  mappedStrides?: readonly number[];
}

export interface TransposeInput {
  A: GPUTensorI;
  permutation: readonly number[];
}

export class TransposeOperation<GPUTensor extends GPUTensorI> extends Operation<
  GPUTensor,
  TransposeInfo,
  TransposeInput
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
    ${this.getVarModifier('mappedStrides')} int mappedStrides[${this.maxRank}];
    `;
  }

  getUniformAttrs(): Input[] {
    return [{name: 'mappedStrides', length: this.maxRank}];
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  getFragmentShader(info: TransposeInfo): string {
    return `
    float process(int index[${this.maxRank}]) {
      return getValueAt(index, mappedStrides, widthA, heightA, A);
    }

    ${this.getDefaultMain()}
    `;
  }

  getTextureNames(): string[] {
    return ['A'];
  }

  calc(input: TransposeInput): GPUTensor {
    if (this.fullyStatic && this.outputShape !== undefined) {
      return this.compute(this.outputShape, {A: input.A});
    }

    const rank = input.A.shape.length;

    const outputShape = this.getOutputShape(input);

    const inputStrides = computeStrides(input.A.shape);
    const mappedStrides = new Array(rank);
    for (let i = 0; i < rank; i++) {
      mappedStrides[i] = inputStrides[input.permutation[i]];
    }

    return this.compute(
      outputShape,
      {A: input.A},
      {mappedStrides: this.pad(mappedStrides)}
    );
  }

  getOutputShape(input: TransposeInput): readonly number[] {
    const rank = input.A.shape.length;

    const outputShape = new Array(rank);
    for (let i = 0; i < rank; i++) {
      outputShape[i] = input.A.shape[input.permutation[i]];
    }

    return outputShape;
  }

  compile(info: TransposeInfo) {
    if (info.shapeA !== undefined) {
      this.maxRank = info.shapeA.length;

      if (info.permutation !== undefined) {
        const rank = info.shapeA.length;

        const inputStrides = computeStrides(info.shapeA);
        const mappedStrides = new Array(rank);
        for (let i = 0; i < rank; i++) {
          mappedStrides[i] = inputStrides[info.permutation[i]];
        }

        info.mappedStrides = mappedStrides;

        delete info['permutation'];
      }
    }

    super.compile(info);
  }

  getCompilationInfo(input: TransposeInput): TransposeInfo {
    const outputShape = this.getOutputShape(input);
    const outputSize = defaultAllocator.getAllocationDimensions(
      getSize(outputShape),
      this.dtype
    );

    const rank = input.A.shape.length;

    const inputStrides = computeStrides(input.A.shape);
    const mappedStrides = new Array(rank);
    for (let i = 0; i < rank; i++) {
      mappedStrides[i] = inputStrides[input.permutation[i]];
    }

    return {
      shapeA: input.A.shape,
      widthA: input.A.memory.width,
      heightA: input.A.memory.height,

      shapeOutput: outputShape,
      widthOutput: outputSize.width,
      heightOutput: outputSize.height,

      mappedStrides,
    };
  }

  getInputInfoString(input: TransposeInput): string {
    return `${input.A.shape}-${input.permutation}`;
  }
}
