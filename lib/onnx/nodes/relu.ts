import {Mode} from '../../model/module';
import {Attributes, Constants} from '../types';
import {ClipNode} from './clip';

export class ReluNode extends ClipNode {
  constructor(
    attributes: Attributes,
    inputs: string[],
    outputs: string[],
    constants: Constants,
    onnxVersion: number,
    mode: Mode
  ) {
    super(attributes, inputs, outputs, constants, onnxVersion, mode);

    this.max = undefined;
    this.min = 0;
  }

  getType() {
    return 'Relu';
  }
}
