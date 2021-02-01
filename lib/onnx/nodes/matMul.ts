import Tensor from '../../types';
import {OnnxNode} from '../node';
import {Attributes, Constants} from '../types';

export class MatMulNode extends OnnxNode {
  constructor(
    attributes: Attributes,
    inputs: string[],
    outputs: string[],
    constants: Constants,
    onnxVersion: number
  ) {
    super(attributes, inputs, outputs, constants, onnxVersion);
  }

  async forward(inputs: Tensor[]): Promise<Tensor[]> {
    const A = inputs[0];
    const B = inputs[1];

    if (this.onnxVersion < 13 && this.onnxVersion >= 9) {
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
