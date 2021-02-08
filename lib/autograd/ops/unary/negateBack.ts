import {Tensor} from '../../../library';
import {BackwardOp, VariableI} from '../../types';

export class NegateBack implements BackwardOp {
  constructor(public input: VariableI) {}

  backward(grad: Tensor): void {
    const gradIn = grad.negate();
    const needed = this.input.backward(gradIn);
    if (!needed) {
      gradIn.delete();
    }
  }

  delete(): void {
    if (!this.input.isLeaf()) {
      this.input.delete();
    }
  }
}
