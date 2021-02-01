import Tensor from '../../types';
import {OnnxNode} from '../node';
import {Attributes, Constants} from '../types';

export class ConcatNode extends OnnxNode {
  private axis?: number;

  constructor(
    attributes: Attributes,
    inputs: string[],
    outputs: string[],
    constants: Constants,
    onnxVersion: number
  ) {
    super(attributes, inputs, outputs, constants, onnxVersion);

    if (onnxVersion < 13) {
      //@ts-ignore
      this.axis = this.getAttributeInt('axis');
    }
  }

  async forward(inputs: Tensor[]): Promise<Tensor[]> {
    if (inputs.length > 2) {
      // This logging seems to slow down the operation more than the operation itself
      //console.warn(`Concat with more than 2 tensors is currently slow. Doing concat with ${inputs.length} tensors`);
    }

    if (this.onnxVersion < 13 && this.axis !== undefined) {
      let result = inputs[0];
      for (let i = 1; i < inputs.length; i++) {
        const newRes = result.concat(inputs[i], this.axis);
        if (i > 1) {
          result.delete();
        }
        result = newRes;
      }

      return [result];
    }
    throw new Error(
      `Concat not implemented for onnx version ${this.onnxVersion}`
    );
  }

  getType() {
    return 'Concat';
  }

  delete(): void {}
}
