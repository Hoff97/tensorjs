import {Mode} from '../../model/module';
import {CPUTensor} from '../../tensor/cpu/tensor';
import Tensor from '../../types';
import {OnnxNode} from '../node';
import {Attributes, Constants} from '../types';

export class ReshapeNode extends OnnxNode {
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
    const x = inputs[0];
    const shape = inputs[1];

    if (!(shape instanceof CPUTensor)) {
      throw new Error('Reshape only works with CPU tensor as shape tensor');
    }

    if (this.onnxVersion < 13) {
      const _shape = new Array(shape.size);
      for (let i = 0; i < shape.size; i++) {
        _shape[i] = shape.get(i);
      }

      return [x.reshape(_shape)];
    }
    throw new Error(
      `Reshape with onnx version ${this.onnxVersion} not yet implemented`
    );
  }

  getType() {
    return 'Reshape';
  }

  delete(): void {}
}
