import {Mode} from '../../model/module';
import {CPUTensor} from '../../tensor/cpu/tensor';
import Tensor from '../../types';
import {OnnxNode} from '../node';
import {Attributes, Constants} from '../types';

export class GatherNode extends OnnxNode {
  private axis: number;

  constructor(
    attributes: Attributes,
    inputs: string[],
    outputs: string[],
    constants: Constants,
    onnxVersion: number,
    mode: Mode
  ) {
    super(attributes, inputs, outputs, constants, onnxVersion, mode);

    this.axis = this.getAttributeInt('axis') || 0;
  }

  async forward(inputs: Tensor<any>[]): Promise<Tensor<any>[]> {
    const x = inputs[0];
    const indices = inputs[1];

    if (!(indices instanceof CPUTensor)) {
      throw new Error('Gather requires CPU tensor for the indices');
    }

    return [x.gather(this.axis, indices)];
  }

  getType() {
    return 'Gather';
  }

  delete(): void {}
}
