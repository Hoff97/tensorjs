import {Mode} from '../../../model/module';
import Tensor, {DType} from '../../../types';
import {Attributes, Constants} from '../../types';
import {BinaryNode} from './binaryNode';

export class DivNode extends BinaryNode {
  constructor(
    attributes: Attributes,
    inputs: string[],
    outputs: string[],
    constants: Constants,
    onnxVersion: number,
    mode: Mode
  ) {
    super(attributes, inputs, outputs, constants, onnxVersion, 'Div', mode);
  }

  compute<DTpe extends DType>(a: Tensor<DTpe>, b: Tensor<DTpe>): Tensor<DTpe> {
    return a.divide(b);
  }
}
