import {
  DTypeGpu,
  GPUTensorConstructor,
  GPUTensorI,
} from '../../../tensor/gpu/interface';
import {GPUMemoryAllocator} from '../../../tensor/gpu/memory';
import {Input} from '../operation';
import {UnaryOperation, UnaryOpInfo, UnaryOpInput} from './unaryOperation';

interface AddMultiplyScalarInfo extends UnaryOpInfo {
  factor?: number;
  add?: number;
}

interface AddMultiplyScalarInput extends UnaryOpInput {
  factor: number;
  add: number;
}

export class AddMultiplyScalarOperation<
  GPUTensor extends GPUTensorI
> extends UnaryOperation<
  GPUTensor,
  AddMultiplyScalarInfo,
  AddMultiplyScalarInput
> {
  constructor(
    tensorConstructor: GPUTensorConstructor<GPUTensor>,
    dtype: DTypeGpu,
    allocator?: GPUMemoryAllocator
  ) {
    super(tensorConstructor, dtype, allocator);
  }

  operation(input: string): string {
    return `factor*${input} + add`;
  }

  calc(input: AddMultiplyScalarInput): GPUTensor {
    return this.compute(
      input.input.shape,
      {X: input.input},
      {factor: input.factor, add: input.add}
    );
  }

  getCompilationInfo(input: AddMultiplyScalarInput): AddMultiplyScalarInfo {
    const info = super.getCompilationInfo(input);

    return {
      ...info,
      factor: input.factor,
      add: input.add,
    };
  }

  getInputInfoString(input: AddMultiplyScalarInput): string {
    return `${super.getInputInfoString(input)}-${input.factor}-${input.add}`;
  }

  getVariables() {
    return `
    ${this.getVarModifier('factor')} float factor;
    ${this.getVarModifier('add')} float add;
    `;
  }

  getUniformAttrs(): Input[] {
    return [
      {name: 'factor', type: 'float'},
      {name: 'add', type: 'float'},
    ];
  }
}
