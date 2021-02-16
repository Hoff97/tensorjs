import {Tensor} from '../../../library';
import {BackwardOp, VariableI} from '../../types';

export class SinBack implements BackwardOp {
  constructor(public input: VariableI) {}

  backward(grad: Tensor): void {
    const cos = this.input.value.cos();
    const gradAbs = grad.multiply(cos);
    cos.delete();
    const needed = this.input.backward(gradAbs);
    if (!needed) {
      gradAbs.delete();
    }
  }

  delete(): void {
    if (!this.input.isLeaf()) {
      this.input.delete();
    }
  }
}
