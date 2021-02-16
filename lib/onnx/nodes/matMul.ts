import {Mode} from '../../model/module';
import Tensor from '../../types';
import {OnnxNode} from '../node';
import {Attributes, Constants} from '../types';

export class MatMulNode extends OnnxNode {
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
    const A = inputs[0];
    const B = inputs[1];

    if (this.onnxVersion < 13) {
      if (A.getShape().length !== B.getShape().length) {
        throw new Error('Automatic broadcasting in MatMul not supported yet');
      }

      return [A.gemm(B)];
    }
    throw new Error(
      `Matmul with onnx version ${this.onnxVersion} not yet implemented`
    );
  }

  getType() {
    return 'MatMul';
  }

  delete(): void {}
}
