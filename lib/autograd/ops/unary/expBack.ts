import {DType, Tensor} from '../../../library';
import {BackwardOp, VariableI} from '../../types';

export class ExpBack<DTpe extends DType> implements BackwardOp<DTpe> {
  constructor(public input: VariableI<DTpe>, public exp: Tensor<DTpe>) {}

  backward(grad: Tensor<DTpe>): void {
    const gradExp = grad.multiply(this.exp);
    const needed = this.input.backward(gradExp);
    if (!needed) {
      gradExp.delete();
    }
  }

  delete(): void {
    if (!this.input.isLeaf()) {
      this.input.delete();
    }
  }
}
