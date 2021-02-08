import {Mode} from '../../model/module';
import {CPUTensor} from '../../tensor/cpu/tensor';
import Tensor from '../../types';
import {getSize} from '../../util/shape';
import {OnnxNode} from '../node';
import {Attributes, Constants} from '../types';
import {createTensor} from '../util';

// This does not support gradients right now, mainly because
// the forward pass needs to directly access the constant value
export class ConstantOfShapeNode extends OnnxNode {
  private tensor?: CPUTensor;

  constructor(
    attributes: Attributes,
    inputs: string[],
    outputs: string[],
    constants: Constants,
    onnxVersion: number,
    mode: Mode
  ) {
    super(attributes, inputs, outputs, constants, onnxVersion, mode);

    if (onnxVersion < 11) {
      const tensor = this.getAttributeTensor('value');
      if (tensor !== null && tensor !== undefined) {
        this.tensor = createTensor(tensor);
      }
    }
  }

  async forward(inputs: Tensor[]): Promise<Tensor[]> {
    const _shape = inputs[0];

    if (this.onnxVersion < 11 && this.tensor !== undefined) {
      if (!(_shape instanceof CPUTensor)) {
        throw new Error('ConstantOfShape needs cpu tensor as shape tensor');
      }
      const shape = new Array(_shape.size);
      for (let i = 0; i < _shape.size; i++) {
        shape[i] = _shape.get(i);
      }

      const size = getSize(shape);
      const values = new Float32Array(size).fill(this.tensor.get(0));

      return [new CPUTensor(shape, values, this.tensor.type)];
    }
    throw new Error(
      `ConstantOfShape not implemented for onnx version ${this.onnxVersion}`
    );
  }

  getType() {
    return 'ConstantOfShape';
  }

  delete(): void {
    if (this.tensor !== undefined) {
      this.tensor.delete();
    }
  }
}
