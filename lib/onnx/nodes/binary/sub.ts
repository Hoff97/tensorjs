import {Mode} from '../../../model/module';
import Tensor, {DType} from '../../../types';
import {Attributes, Constants} from '../../types';
import {BinaryNode} from './binaryNode';

export class SubNode extends BinaryNode {
  constructor(
    attributes: Attributes,
    inputs: string[],
    outputs: string[],
    constants: Constants,
    onnxVersion: number,
    mode: Mode
  ) {
    super(attributes, inputs, outputs, constants, onnxVersion, 'Sub', mode);
  }

  compute<DTpe extends DType>(a: Tensor<DTpe>, b: Tensor<DTpe>): Tensor<DTpe> {
    return a.subtract(b);
  }
}
