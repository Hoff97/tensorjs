import {Mode} from '../../model/module';
import {CPUTensor} from '../../tensor/cpu/tensor';
import Tensor from '../../types';
import {OnnxNode} from '../node';
import {Attributes, Constants} from '../types';

export class ShapeNode extends OnnxNode {
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
    if (this.onnxVersion < 13) {
      const a = inputs[0];

      const shape = a.getShape();

      return [new CPUTensor([shape.length], [...shape], 'int')];
    }
    throw new Error(
      `Shape not implemented for onnx version ${this.onnxVersion}`
    );
  }

  getType() {
    return 'Shape';
  }

  delete(): void {}
}
