import {DType, Tensor} from '../../../library';
import {BackwardOp, VariableI} from '../../types';

export class SqrtBack<DTpe extends DType> implements BackwardOp<DTpe> {
  constructor(public input: VariableI<DTpe>, public sqrt: Tensor<DTpe>) {}

  backward(grad: Tensor<DTpe>): void {
    const gradSqrt = grad.divide(this.sqrt, 0.5);
    const needed = this.input.backward(gradSqrt);
    if (!needed) {
      gradSqrt.delete();
    }
  }

  delete(): void {
    if (!this.input.isLeaf()) {
      this.input.delete();
    }
  }
}
