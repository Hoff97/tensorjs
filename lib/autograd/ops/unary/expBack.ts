import {Tensor} from '../../../library';
import {BackwardOp, VariableI} from '../../types';

export class ExpBack implements BackwardOp {
  constructor(public input: VariableI, public exp: Tensor) {}

  backward(grad: Tensor): void {
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
