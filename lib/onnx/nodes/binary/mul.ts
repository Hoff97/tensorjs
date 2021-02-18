import {Mode} from '../../../model/module';
import Tensor from '../../../types';
import {Attributes, Constants} from '../../types';
import {BinaryNode} from './binaryNode';

export class MulNode extends BinaryNode {
  constructor(
    attributes: Attributes,
    inputs: string[],
    outputs: string[],
    constants: Constants,
    onnxVersion: number,
    mode: Mode
  ) {
    super(attributes, inputs, outputs, constants, onnxVersion, 'Mul', mode);
  }

  compute(a: Tensor, b: Tensor): Tensor {
    return a.multiply(b);
  }
}
