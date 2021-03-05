import {
  DTypeGpu,
  GPUTensorConstructor,
  GPUTensorI,
} from '../../../tensor/gpu/interface';
import {GPUMemoryAllocator} from '../../../tensor/gpu/memory';
import {Input} from '../operation';
import {BinaryOperation, BinaryOpInfo, BinaryOpInput} from './binaryOperation';

export interface AddInfo extends BinaryOpInfo {
  alpha?: number;
  beta?: number;
}

export interface AddInput extends BinaryOpInput {
  alpha: number;
  beta: number;
}

export class AddOperation<GPUTensor extends GPUTensorI> extends BinaryOperation<
  GPUTensor,
  AddInfo,
  AddInput
> {
  constructor(
    tensorConstructor: GPUTensorConstructor<GPUTensor>,
    dtype: DTypeGpu,
    allocator?: GPUMemoryAllocator
  ) {
    super(tensorConstructor, dtype, allocator);
  }

  getOp(a: string, b: string): string {
    return `alpha*${a} + beta*${b}`;
  }

  calc(input: AddInput): GPUTensor {
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

  getCompilationInfo(input: AddInput): AddInfo {
    const info = super.getCompilationInfo(input);
    return {
      ...info,
      alpha: input.alpha,
      beta: input.beta,
    };
  }

  getInputInfoString(input: AddInput): string {
    return `${super.getInputInfoString(input)}-${input.alpha}-${input.beta}`;
  }
}
