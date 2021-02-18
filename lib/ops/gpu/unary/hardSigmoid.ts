import {GPUTensorConstructor, GPUTensorI} from '../../../tensor/gpu/interface';
import {GPUMemoryAllocator} from '../../../tensor/gpu/memory';
import {Precision} from '../../../types';
import {Input} from '../operation';
import {UnaryOperation, UnaryOpInfo, UnaryOpInput} from './unaryOperation';

interface HardSigmoidInfo extends UnaryOpInfo {
  alpha?: number;
  beta?: number;
}

interface HardSigmoidInput extends UnaryOpInput {
  alpha: number;
  beta: number;
}

export class HardSigmoidOperation<
  GPUTensor extends GPUTensorI
> extends UnaryOperation<GPUTensor, HardSigmoidInfo, HardSigmoidInput> {
  constructor(
    tensorConstructor: GPUTensorConstructor<GPUTensor>,
    allocator?: GPUMemoryAllocator
  ) {
    super(tensorConstructor, allocator);
  }

  operation(input: string): string {
    return `max(vec4(0.0,0.0,0.0,0.0), min(vec4(1.0,1.0,1.0,1.0), alpha*${input} + beta))`;
  }

  calc(input: HardSigmoidInput): GPUTensor {
    return this.compute(
      input.input.shape,
      {X: input.input},
      {alpha: input.alpha, beta: input.beta}
    );
  }

  getCompilationInfo(
    input: HardSigmoidInput,
    precision: Precision
  ): HardSigmoidInfo {
    const info = super.getCompilationInfo(input, precision);

    return {
      ...info,
      alpha: input.alpha,
      beta: input.beta,
    };
  }

  getInputInfoString(input: HardSigmoidInput): string {
    return `${super.getInputInfoString(input)}-${input.alpha}-${input.beta}`;
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
}
