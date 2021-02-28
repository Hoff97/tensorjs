import {Mode} from '../../model/module';
import {CPUTensor} from '../../tensor/cpu/tensor';
import Tensor from '../../types';
import {OnnxNode} from '../node';
import {Attributes, Constants} from '../types';

export class CastNode extends OnnxNode {
  private to: string;

  constructor(
    attributes: Attributes,
    inputs: string[],
    outputs: string[],
    constants: Constants,
    onnxVersion: number,
    mode: Mode
  ) {
    super(attributes, inputs, outputs, constants, onnxVersion, mode);

    //@ts-ignore
    this.to = this.getAttributeString('to');
  }

  async forward(inputs: Tensor<any>[]): Promise<Tensor<any>[]> {
    const x = inputs[0];

    // TODO: Convert to to correct dtype
    return [x.cast(this.to as any)];
  }

  getType() {
    return 'Cast';
  }

  delete(): void {}
}
