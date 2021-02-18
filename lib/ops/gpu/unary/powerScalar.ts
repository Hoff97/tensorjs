import {GPUTensorConstructor, GPUTensorI} from '../../../tensor/gpu/interface';
import {GPUMemoryAllocator} from '../../../tensor/gpu/memory';
import {Precision} from '../../../types';
import {Input} from '../operation';
import {UnaryOperation, UnaryOpInfo, UnaryOpInput} from './unaryOperation';

interface PowerScalarInfo extends UnaryOpInfo {
  factor?: number;
  power?: number;
}

interface PowerScalarInput extends UnaryOpInput {
  factor: number;
  power: number;
}

export class PowerScalarOperation<
  GPUTensor extends GPUTensorI
> extends UnaryOperation<GPUTensor, PowerScalarInfo, PowerScalarInput> {
  constructor(
    tensorConstructor: GPUTensorConstructor<GPUTensor>,
    allocator?: GPUMemoryAllocator
  ) {
    super(tensorConstructor, allocator);
  }

  operation(input: string): string {
    throw new Error('Method not implemented.');
  }

  getFragmentShader(info: PowerScalarInfo): string {
    return `
    void main() {
      initVars();

      if (power < 0.0) {
        gl_FragColor = vec4(factor,factor,factor,factor) / pow(texture2D(X, uv), vec4(-power,-power,-power,-power));
      } else {
        gl_FragColor = pow(texture2D(X, uv), vec4(power,power,power,power)) * factor;
      }
    }
    `;
  }

  calc(input: PowerScalarInput): GPUTensor {
    return this.compute(
      input.input.shape,
      {X: input.input},
      {factor: input.factor, power: input.power}
    );
  }

  getCompilationInfo(
    input: PowerScalarInput,
    precision: Precision
  ): PowerScalarInfo {
    const info = super.getCompilationInfo(input, precision);

    return {
      ...info,
      factor: input.factor,
      power: input.power,
    };
  }

  getInputInfoString(input: PowerScalarInput): string {
    return `${super.getInputInfoString(input)}-${input.factor}-${input.power}`;
  }

  getVariables() {
    return `
    ${this.getVarModifier('factor')} float factor;
    ${this.getVarModifier('power')} float power;
    `;
  }

  getUniformAttrs(): Input[] {
    return [
      {name: 'factor', type: 'float'},
      {name: 'power', type: 'float'},
    ];
  }
}
