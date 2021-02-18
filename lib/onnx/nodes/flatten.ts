import {Mode} from '../../model/module';
import Tensor from '../../types';
import {OnnxNode} from '../node';
import {Attributes, Constants} from '../types';

export class FlattenNode extends OnnxNode {
  private axis?: number;

  constructor(
    attributes: Attributes,
    inputs: string[],
    outputs: string[],
    constants: Constants,
    onnxVersion: number,
    mode: Mode
  ) {
    super(attributes, inputs, outputs, constants, onnxVersion, mode);

    const axis = this.getAttributeInt('axis');
    if (axis !== null) {
      this.axis = axis;
    }
  }

  async forward(inputs: Tensor[]): Promise<Tensor[]> {
    const tensor = inputs[0];

    return [tensor.flatten(this.axis)];
  }

  getType() {
    return 'Flatten';
  }

  delete(): void {}
}
