import {Mode} from '../../model/module';
import Tensor from '../../types';
import {OnnxNode} from '../node';
import {Attributes, Constants} from '../types';

export class PReluNode extends OnnxNode {
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
    const tensor = inputs[0];
    const slope = inputs[1];

    const below = tensor.clip(undefined, 0);
    const above = tensor.clip(0, undefined);

    const belowRes = below.multiply(slope);
    below.delete();

    const result = belowRes.add(above);

    belowRes.delete();
    above.delete();
    return [result];
  }

  delete(): void {}

  getType() {
    return 'PRElu';
  }
}
