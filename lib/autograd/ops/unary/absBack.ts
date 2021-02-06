import {Tensor} from '../../../library';
import {BackwardOp, VariableI} from '../../types';

export class AbsBack implements BackwardOp {
  constructor(public input: VariableI) {}

  backward(grad: Tensor): void {
    const sign = this.input.value.sign();
    const gradAbs = grad.multiply(sign);
    sign.delete();
    this.input.backward(gradAbs);
  }

  delete(): void {
    if (!this.input.isLeaf()) {
      this.input.delete();
    }
  }
}
