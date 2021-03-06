import {Mode} from '../../../model/module';
import Tensor, {DType} from '../../../types';
import {OnnxNode} from '../../node';
import {Attributes, Constants} from '../../types';

export class MeanNode extends OnnxNode {
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

  async forward<DTpe extends DType>(
    inputs: Tensor<DTpe>[]
  ): Promise<Tensor<DTpe>[]> {
    if (inputs.length > 2) {
      // This logging seems to slow down the operation more than the operation itself
      //console.warn(`Sum with more than 2 tensors is currently slow. Doing concat with ${inputs.length} tensors`);
    }

    let result = inputs[0];
    for (let i = 1; i < inputs.length; i++) {
      const newRes = result.add(inputs[i]);
      if (i > 1) {
        result.delete();
      }
      result = newRes;
    }

    const mean = result.multiplyScalar(1 / inputs.length);
    if (inputs.length > 2) {
      result.delete();
    }

    return [mean];
  }

  getType() {
    return 'Mean';
  }

  delete(): void {}
}
