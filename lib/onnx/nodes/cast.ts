import {cast} from '../../ops/cpu/cast';
import {CPUTensor} from '../../tensor/cpu/tensor';
import Tensor from '../../types';
import {OnnxNode} from '../node';
import {Attributes, Constants} from '../types';

export class CastNode extends OnnxNode {
  private to: string;

  constructor(
    attributes: Attributes,
    inputs: string[],
    outputs: string[],
    constants: Constants,
    onnxVersion: number
  ) {
    super(attributes, inputs, outputs, constants, onnxVersion);

    this.to = this.getAttributeString('to');
  }

  async forward(inputs: Tensor[]): Promise<Tensor[]> {
    const x = inputs[0];

    if (x instanceof CPUTensor) {
      return [cast(x, this.to)];
    }
    throw new Error('Can only cast CPU tensors right now');
  }

  getType() {
    return 'Cast';
  }

  delete(): void {}
}
