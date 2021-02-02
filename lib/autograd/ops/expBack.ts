import {Tensor} from '../../library';
import {BackwardOp, VariableI} from '../types';

export class ExpBack implements BackwardOp {
  constructor(public input: VariableI | Tensor, public exp: Tensor) {}

  backward(grad: Tensor): void {
    if ((this.input as VariableI).backward !== undefined) {
      const gradExp = grad.multiply(this.exp);
      (this.input as VariableI).backward(gradExp);
    }
  }
}
