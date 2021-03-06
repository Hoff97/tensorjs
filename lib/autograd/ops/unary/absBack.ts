import {DType, Tensor} from '../../../library';
import {BackwardOp, VariableI} from '../../types';

export class AbsBack<DTpe extends DType> implements BackwardOp<DTpe> {
  constructor(public input: VariableI<DTpe>) {}

  backward(grad: Tensor<DTpe>): void {
    const sign = this.input.value.sign();
    const gradAbs = grad.multiply(sign);
    sign.delete();
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
