import {Mode} from '../../../model/module';
import Tensor, {DType} from '../../../types';
import {Attributes, Constants} from '../../types';
import {UnaryNode} from './unaryNode';

export class HardSigmoidNode extends UnaryNode {
  alpha: number;
  beta: number;

  constructor(
    attributes: Attributes,
    inputs: string[],
    outputs: string[],
    constants: Constants,
    onnxVersion: number,
    mode: Mode
  ) {
    super(
      attributes,
      inputs,
      outputs,
      constants,
      onnxVersion,
      'HardSigmoid',
      mode
    );

    this.alpha = this.getAttributeFloat('alpha') || 0.2;
    this.beta = this.getAttributeFloat('beta') || 0.5;
  }

  compute<DTpe extends DType>(x: Tensor<DTpe>): Tensor<DTpe> {
    return x.hardSigmoid(this.alpha, this.beta);
  }
}
