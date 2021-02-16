import {Tensor} from '../../../library';
import {BackwardOp, VariableI} from '../../types';

export class TanBack implements BackwardOp {
  constructor(public input: VariableI) {}

  backward(grad: Tensor): void {
    const cos = this.input.value.cos();
    const cos2 = cos.multiply(cos);
    cos.delete();
    const gradTan = grad.divide(cos2);
    cos2.delete();
    const needed = this.input.backward(gradTan);
    if (!needed) {
      gradTan.delete();
    }
  }

  delete(): void {
    if (!this.input.isLeaf()) {
      this.input.delete();
    }
  }
}
