import {Tensor} from '../../../library';
import {BackwardOp, VariableI} from '../../types';

export class LogBack implements BackwardOp {
  constructor(public input: VariableI) {}

  backward(grad: Tensor): void {
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
