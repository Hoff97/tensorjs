import {Mode} from '../../../model/module';
import Tensor from '../../../types';
import {Attributes, Constants} from '../../types';
import {UnaryNode} from './unaryNode';

export class SqrtNode extends UnaryNode {
  constructor(
    attributes: Attributes,
    inputs: string[],
    outputs: string[],
    constants: Constants,
    onnxVersion: number,
    mode: Mode
  ) {
    super(attributes, inputs, outputs, constants, onnxVersion, 'Sqrt', mode);
  }

  compute(x: Tensor): Tensor {
    return x.sqrt();
  }
}
