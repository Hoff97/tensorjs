import {Tensor} from '../../../library';
import {BackwardOp, VariableI} from '../../types';

export class CosBack implements BackwardOp {
  constructor(public input: VariableI) {}

  backward(grad: Tensor): void {
    const sin = this.input.value.sin();
    const gradAbs = grad.multiply(sin, -1);
    sin.delete();
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
