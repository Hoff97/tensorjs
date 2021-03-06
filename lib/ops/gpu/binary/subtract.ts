import {
  DTypeGpu,
  GPUTensorConstructor,
  GPUTensorI,
} from '../../../tensor/gpu/interface';
import {GPUMemoryAllocator} from '../../../tensor/gpu/memory';
import {Input} from '../operation';
import {BinaryOperation, BinaryOpInfo, BinaryOpInput} from './binaryOperation';

export interface SubtractInfo extends BinaryOpInfo {
  alpha?: number;
  beta?: number;
}

export interface SubtractInput extends BinaryOpInput {
  alpha: number;
  beta: number;
}

export class SubtractOperation<
  GPUTensor extends GPUTensorI
> extends BinaryOperation<GPUTensor, SubtractInfo, SubtractInput> {
  constructor(
    tensorConstructor: GPUTensorConstructor<GPUTensor>,
    dtype: DTypeGpu,
    allocator?: GPUMemoryAllocator
  ) {
    super(tensorConstructor, dtype, allocator);
  }

  getOp(a: string, b: string): string {
    return `alpha*${a} - beta*${b}`;
  }

  calc(input: SubtractInput): GPUTensor {
    return this.compute(
      input.outputShape,
      {A: input.A, B: input.B},
      {alpha: input.alpha, beta: input.beta}
    );
  }

  getVariables() {
    return `
    ${this.getVarModifier('alpha')} float alpha;
    ${this.getVarModifier('beta')} float beta;
    `;
  }

  getUniformAttrs(): Input[] {
    return [
      {name: 'alpha', type: 'float'},
      {name: 'beta', type: 'float'},
    ];
  }

  getCompilationInfo(input: SubtractInput): SubtractInfo {
    const info = super.getCompilationInfo(input);
    return {
      ...info,
      alpha: input.alpha,
      beta: input.beta,
    };
  }

  getInputInfoString(input: SubtractInput): string {
    return `${super.getInputInfoString(input)}-${input.alpha}-${input.beta}`;
  }
}
