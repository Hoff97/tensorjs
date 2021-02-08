import {Tensor} from '../../../library';
import {BackwardOp, VariableI} from '../../types';

export class SqrtBack implements BackwardOp {
  constructor(public input: VariableI, public sqrt: Tensor) {}

  backward(grad: Tensor): void {
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
