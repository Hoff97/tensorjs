import {Mode} from '../../model/module';
import {CPUTensor} from '../../tensor/cpu/tensor';
import {GPUTensor} from '../../tensor/gpu/tensor';
import {WASMTensor} from '../../tensor/wasm/tensor';
import Tensor, {Precision} from '../../types';
import {Backend} from '../../util/convert';
import {OnnxNode} from '../node';
import {Attributes, Constants} from '../types';

export class RangeNode extends OnnxNode {
  backend: Backend = 'CPU';
  precision?: Precision;

  constructor(
    attributes: Attributes,
    inputs: string[],
    outputs: string[],
    constants: Constants,
    onnxVersion: number,
    mode: Mode
  ) {
    super(attributes, inputs, outputs, constants, onnxVersion, mode);
  }

  async forward(inputs: Tensor[]): Promise<Tensor[]> {
    const start = inputs[0];
    const limit = inputs[1];
    const delta = inputs[2];

    const startValue = (await this.toValues(start))[0];
    const limitValue = (await this.toValues(limit))[0];
    const deltaValue = (await this.toValues(delta))[0];

    if (this.backend === 'CPU') {
      return [CPUTensor.range(startValue, limitValue, deltaValue)];
    } else if (this.backend === 'WASM') {
      return [WASMTensor.range(startValue, limitValue, deltaValue)];
    } else {
      return [
        GPUTensor.range(startValue, limitValue, deltaValue, this.precision),
      ];
    }
  }

  delete(): void {}

  getType() {
    return 'Range';
  }

  async toCPU() {
    this.backend = 'CPU';
  }
  async toWASM() {
    this.backend = 'WASM';
  }
  async toGPU(precision: Precision) {
    this.backend = 'GPU';
    this.precision = precision;
  }
}
