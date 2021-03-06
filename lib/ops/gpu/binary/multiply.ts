import {
  DTypeGpu,
  GPUTensorConstructor,
  GPUTensorI,
} from '../../../tensor/gpu/interface';
import {GPUMemoryAllocator} from '../../../tensor/gpu/memory';
import {Input} from '../operation';
import {BinaryOperation, BinaryOpInfo, BinaryOpInput} from './binaryOperation';

export interface MultiplyInfo extends BinaryOpInfo {
  alpha?: number;
}

export interface MultiplyInput extends BinaryOpInput {
  alpha: number;
}

export class MultiplyOperation<
  GPUTensor extends GPUTensorI
> extends BinaryOperation<GPUTensor, MultiplyInfo, MultiplyInput> {
  constructor(
    tensorConstructor: GPUTensorConstructor<GPUTensor>,
    dtype: DTypeGpu,
    allocator?: GPUMemoryAllocator
  ) {
    super(tensorConstructor, dtype, allocator);
  }

  getOp(a: string, b: string): string {
    return `alpha*${a} * ${b}`;
  }

  calc(input: MultiplyInput): GPUTensor {
    return this.compute(
      input.outputShape,
      {A: input.A, B: input.B},
      {alpha: input.alpha}
    );
  }

  getVariables() {
    return `
    ${this.getVarModifier('alpha')} float alpha;
    `;
  }

  getUniformAttrs(): Input[] {
    return [{name: 'alpha', type: 'float'}];
  }

  getCompilationInfo(input: MultiplyInput): MultiplyInfo {
    const info = super.getCompilationInfo(input);
    return {
      ...info,
      alpha: input.alpha,
    };
  }

  getInputInfoString(input: MultiplyInput): string {
    return `${super.getInputInfoString(input)}-${input.alpha}`;
  }
}
