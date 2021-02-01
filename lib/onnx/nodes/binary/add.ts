import Tensor from '../../../types';
import {Attributes, Constants} from '../../types';
import {BinaryNode} from './binaryNode';

export class AddNode extends BinaryNode {
  constructor(
    attributes: Attributes,
    inputs: string[],
    outputs: string[],
    constants: Constants,
    onnxVersion: number
  ) {
    super(attributes, inputs, outputs, constants, onnxVersion, 'Add');
  }

  compute(a: Tensor, b: Tensor): Tensor {
    return a.add(b);
  }
}
