import {Tensor} from '../../library';
import {BackwardOp, VariableI} from '../types';

export class LogBack implements BackwardOp {
  constructor(public input: VariableI | Tensor) {}

  backward(grad: Tensor): void {
    if (
      (this.input as VariableI).backward !== undefined &&
      (this.input as VariableI).value !== undefined
    ) {
      const gradExp = grad.divide((this.input as VariableI).value);
      (this.input as VariableI).backward(gradExp);
    }
  }
}
