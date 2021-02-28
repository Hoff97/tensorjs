import {Mode} from '../../model/module';
import Tensor, {DType} from '../../types';
import {OnnxNode} from '../node';
import {Attributes, Constants} from '../types';

export class SeluNode extends OnnxNode {
  alpha: number;
  gamma: number;

  constructor(
    attributes: Attributes,
    inputs: string[],
    outputs: string[],
    constants: Constants,
    onnxVersion: number,
    mode: Mode
  ) {
    super(attributes, inputs, outputs, constants, onnxVersion, mode);

    this.alpha = this.getAttributeFloat('alpha') || 1.67326319217681884765625;
    this.gamma = this.getAttributeFloat('gamma') || 1.05070102214813232421875;
  }

  async forward<DTpe extends DType>(
    inputs: Tensor<DTpe>[]
  ): Promise<Tensor<DTpe>[]> {
    const tensor = inputs[0];

    const below = tensor.clip(undefined, 0);
    const above = tensor.clip(0, undefined);

    const belowExp = below.exp();
    below.delete();
    const b = belowExp.addMultiplyScalar(
      this.gamma * this.alpha,
      -this.alpha * this.gamma
    );
    belowExp.delete();

    const result = b.add(above, 1.0, this.gamma);

    b.delete();
    above.delete();
    return [result];
  }

  delete(): void {}

  getType() {
    return 'Selu';
  }
}
