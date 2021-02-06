import {Tensor} from '../../../library';
import {BackwardOp, VariableI} from '../../types';

export class LogBack implements BackwardOp {
  constructor(public input: VariableI) {}

  backward(grad: Tensor): void {
    const gradLog = grad.divide(this.input.value);
    this.input.backward(gradLog);
  }

  delete(): void {
    if (!this.input.isLeaf()) {
      this.input.delete();
    }
  }
}
