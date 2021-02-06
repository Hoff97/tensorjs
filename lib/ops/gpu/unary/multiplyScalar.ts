import {GPUTensorConstructor, GPUTensorI} from '../../../tensor/gpu/interface';
import {GPUMemoryAllocator} from '../../../tensor/gpu/memory';
import {Precision} from '../../../types';
import {Input} from '../operation';
import {UnaryOperation, UnaryOpInfo, UnaryOpInput} from './unaryOperation';

interface MultiplyScalarInfo extends UnaryOpInfo {
  scalar?: number;
}

interface MultiplyScalarInput extends UnaryOpInput {
  scalar: number;
}

export class MultiplyScalarOperation<
  GPUTensor extends GPUTensorI
> extends UnaryOperation<GPUTensor, MultiplyScalarInfo, MultiplyScalarInput> {
  constructor(
    tensorConstructor: GPUTensorConstructor<GPUTensor>,
    allocator?: GPUMemoryAllocator
  ) {
    super(tensorConstructor, allocator);
  }

  operation(input: string): string {
    return `scalar*${input}`;
  }

  calc(input: MultiplyScalarInput): GPUTensor {
    return this.compute(
      input.input.shape,
      {X: input.input},
      {scalar: input.scalar}
    );
  }

  getCompilationInfo(
    input: MultiplyScalarInput,
    precision: Precision
  ): MultiplyScalarInfo {
    const info = super.getCompilationInfo(input, precision);

    return {
      ...info,
      scalar: input.scalar,
    };
  }

  getInputInfoString(input: MultiplyScalarInput): string {
    return `${super.getInputInfoString(input)}-${input.scalar}`;
  }

  getVariables() {
    return `
    ${this.getVarModifier('scalar')} float scalar;
    `;
  }

  getUniformAttrs(): Input[] {
    return [{name: 'scalar', type: 'float'}];
  }
}
