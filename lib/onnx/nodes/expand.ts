import {Mode} from '../../model/module';
import {CPUTensor} from '../../tensor/cpu/tensor';
import Tensor from '../../types';
import {OnnxNode} from '../node';
import {Attributes, Constants} from '../types';

export class ExpandNode extends OnnxNode {
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

  async forward(inputs: Tensor<any>[]): Promise<Tensor<any>[]> {
    if (this.onnxVersion < 13) {
      const tensor = inputs[0];

      const _shape = inputs[1];
      if (!(_shape instanceof CPUTensor)) {
        throw new Error('Expand needs cpu tensor as shape tensor');
      }
      const shape = new Array(_shape.size);
      for (let i = 0; i < _shape.size; i++) {
        shape[i] = _shape.get(i);
      }

      return [tensor.expand(shape)];
    }
    throw new Error(
      `Expand not yet implemented for onnx version ${this.onnxVersion}`
    );
  }

  getType() {
    return 'Expand';
  }

  delete(): void {}
}
