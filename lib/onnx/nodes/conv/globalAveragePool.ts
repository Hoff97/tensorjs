import {Mode} from '../../../model/module';
import Tensor from '../../../types';
import {OnnxNode} from '../../node';
import {Attributes, Constants} from '../../types';

export class GlobalAveragePoolNode extends OnnxNode {
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
    const x = inputs[0];

    const axes = new Array(x.getShape().length - 2);
    for (let i = 0; i < x.getShape().length - 2; i++) {
      axes[i] = i + 2;
    }

    return [x.reduceMean(axes, true)];
  }

  getType() {
    return 'GlobalAveragePool';
  }

  delete(): void {}
}
