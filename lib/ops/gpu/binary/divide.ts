import {GPUTensorConstructor, GPUTensorI} from '../../../tensor/gpu/interface';
import {GPUMemoryAllocator} from '../../../tensor/gpu/memory';
import {Precision} from '../../../types';
import {Input} from '../operation';
import {BinaryOperation, BinaryOpInfo, BinaryOpInput} from './binaryOperation';

export interface DivideInfo extends BinaryOpInfo {
  alpha?: number;
}

export interface DivideInput extends BinaryOpInput {
  alpha: number;
}

export class DivideOperation<
  GPUTensor extends GPUTensorI
> extends BinaryOperation<GPUTensor, DivideInfo, DivideInput> {
  constructor(
    tensorConstructor: GPUTensorConstructor<GPUTensor>,
    allocator?: GPUMemoryAllocator
  ) {
    super(tensorConstructor, allocator);
  }

  getOp(a: string, b: string): string {
    return `alpha*${a} / ${b}`;
  }

  calc(input: DivideInput): GPUTensor {
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

  getCompilationInfo(input: DivideInput, precision: Precision): DivideInfo {
    const info = super.getCompilationInfo(input, precision);
    return {
      ...info,
      alpha: input.alpha,
    };
  }

  getInputInfoString(input: DivideInput): string {
    return `${super.getInputInfoString(input)}-${input.alpha}`;
  }
}