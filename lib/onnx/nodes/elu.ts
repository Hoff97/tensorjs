import {Mode} from '../../model/module';
import Tensor from '../../types';
import {OnnxNode} from '../node';
import {Attributes, Constants} from '../types';

export class EluNode extends OnnxNode {
  alpha: number;

  constructor(
    attributes: Attributes,
    inputs: string[],
    outputs: string[],
    constants: Constants,
    onnxVersion: number,
    mode: Mode
  ) {
    super(attributes, inputs, outputs, constants, onnxVersion, mode);

    this.alpha = this.getAttributeFloat('alpha') || 1.0;
  }

  async forward(inputs: Tensor[]): Promise<Tensor[]> {
    const tensor = inputs[0];

    const below = tensor.clip(undefined, 0);
    const above = tensor.clip(0, undefined);

    const belowExp = below.exp();
    below.delete();
    const b = belowExp.addMultiplyScalar(this.alpha, -this.alpha);
    belowExp.delete();

    const result = b.add(above);

    b.delete();
    above.delete();
    return [result];
  }

  delete(): void {}

  getType() {
    return 'Elu';
  }
}
