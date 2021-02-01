import Tensor from '../../../types';
import {Attributes, Constants} from '../../types';
import {UnaryNode} from './unaryNode';

export class CeilNode extends UnaryNode {
  constructor(
    attributes: Attributes,
    inputs: string[],
    outputs: string[],
    constants: Constants,
    onnxVersion: number
  ) {
    super(attributes, inputs, outputs, constants, onnxVersion, 'Ceil');
  }

  compute(x: Tensor): Tensor {
    return x.ceil();
  }
}
