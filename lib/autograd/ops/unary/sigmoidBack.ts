import {DType, Tensor} from '../../../library';
import {BackwardOp, VariableI} from '../../types';

export class SigmoidBack<DTpe extends DType> implements BackwardOp<DTpe> {
  constructor(public input: VariableI<DTpe>, public sigmoid: Tensor<DTpe>) {}

  backward(grad: Tensor<DTpe>): void {
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
