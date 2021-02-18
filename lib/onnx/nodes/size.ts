import {Mode} from '../../model/module';
import {CPUTensor} from '../../tensor/cpu/tensor';
import Tensor from '../../types';
import {getSize} from '../../util/shape';
import {OnnxNode} from '../node';
import {Attributes, Constants} from '../types';

export class SizeNode extends OnnxNode {
  constructor(
    attributes: Attributes,
    inputs: string[],
    outputs: string[],
    constants: Constants,
    onnxVersion: number,
    mode: Mode
  ) {
    super(attributes, inputs, outputs, constants, onnxVersion, mode);
  }

  async forward(inputs: Tensor[]): Promise<Tensor[]> {
    const a = inputs[0];

    const shape = a.getShape();
    const size = getSize(shape);

    return [new CPUTensor([1], [size], 'int')];
  }

  getType() {
    return 'Size';
  }

  delete(): void {}
}
