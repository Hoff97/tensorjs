import {Mode} from '../../model/module';
import Tensor, {DType} from '../../types';
import {OnnxNode} from '../node';
import {Attributes, Constants} from '../types';

export class CeluNode extends OnnxNode {
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

  async forward<DTpe extends DType>(
    inputs: Tensor<DTpe>[]
  ): Promise<Tensor<DTpe>[]> {
    const tensor = inputs[0];

    const below = tensor.clip(undefined, 0);
    const above = tensor.clip(0, undefined);

    // max(0,x) + min(0,alpha*(exp(x/alpha)-1))

    const belowDiv = below.multiplyScalar(1 / this.alpha);
    below.delete();
    const belowExp = belowDiv.exp();
    belowDiv.delete();
    const belowRes = belowExp.addMultiplyScalar(this.alpha, -this.alpha);
    belowExp.delete();

    const result = belowRes.add(above);

    belowRes.delete();
    above.delete();
    return [result];
  }

  delete(): void {}

  getType() {
    return 'Celu';
  }
}
