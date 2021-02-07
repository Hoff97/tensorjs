import {Tensor} from '../../../library';
import {BackwardOp, VariableI} from '../../types';

export class SigmoidBack implements BackwardOp {
  constructor(public input: VariableI, public sigmoid: Tensor) {}

  backward(grad: Tensor): void {
    const oneMinus = this.sigmoid.addMultiplyScalar(-1, 1);
    const mult = this.sigmoid.multiply(oneMinus);
    oneMinus.delete();
    const gradIn = mult.multiply(grad);
    mult.delete();
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
