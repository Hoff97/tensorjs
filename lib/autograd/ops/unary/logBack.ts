import {DType, Tensor} from '../../../library';
import {BackwardOp, VariableI} from '../../types';

export class LogBack<DTpe extends DType> implements BackwardOp<DTpe> {
  constructor(public input: VariableI<DTpe>) {}

  backward(grad: Tensor<DTpe>): void {
    const gradLog = grad.divide(this.input.value);
    const needed = this.input.backward(gradLog);
    if (!needed) {
      gradLog.delete();
    }
  }

  delete(): void {
    if (!this.input.isLeaf()) {
      this.input.delete();
    }
  }
}
