import {Mode} from '../../model/module';
import Tensor, {DType} from '../../types';
import {OnnxNode} from '../node';
import {Attributes, Constants} from '../types';

export class GemmNode extends OnnxNode {
  private alpha: number;
  private beta: number;
  private transA: boolean;
  private transB: boolean;

  constructor(
    attributes: Attributes,
    inputs: string[],
    outputs: string[],
    constants: Constants,
    onnxVersion: number,
    mode: Mode
  ) {
    super(attributes, inputs, outputs, constants, onnxVersion, mode);

    this.alpha = this.getAttributeFloat('alpha') || 1.0;
    this.beta = this.getAttributeFloat('beta') || 1.0;

    const transA = this.getAttributeInt('transA');
    const transB = this.getAttributeInt('transB');

    this.transA = transA === 1;
    this.transB = transB === 1;
  }

  async forward<DTpe extends DType>(
    inputs: Tensor<DTpe>[]
  ): Promise<Tensor<DTpe>[]> {
    const a = inputs[0];
    const b = inputs[1];
    const c = inputs[2];

    return [a.gemm(b, this.transA, this.transB, this.alpha, c, this.beta)];
  }

  getType() {
    return 'Gemm';
  }

  delete(): void {}
}
