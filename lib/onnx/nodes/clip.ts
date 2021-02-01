import Tensor from '../../types';
import {OnnxNode} from '../node';
import {Attributes, Constants} from '../types';

export class ClipNode extends OnnxNode {
  public min?: number;
  public max?: number;

  constructor(
    attributes: Attributes,
    inputs: string[],
    outputs: string[],
    constants: Constants,
    onnxVersion: number
  ) {
    super(attributes, inputs, outputs, constants, onnxVersion);

    if (onnxVersion < 11) {
      //@ts-ignore
      this.min = this.getAttributeFloat('min');
      //@ts-ignore
      this.max = this.getAttributeFloat('max');
    }
  }

  async forward(inputs: Tensor[]): Promise<Tensor[]> {
    const x = inputs[0];

    if (this.onnxVersion < 11) {
      return [x.clip(this.min, this.max)];
    } else {
      const min = inputs.length > 1 ? inputs[1] : undefined;
      const max = inputs.length > 2 ? inputs[2] : undefined;
      if (min === undefined && max === undefined) {
        return [x.copy()];
      }
      throw new Error('Clip with onnx version >= 11 not yet implemented');
    }
  }

  getType() {
    return 'Clip';
  }

  delete(): void {}
}
