import {Tensor} from '../../library';
import {BackwardOp, VariableI} from '../types';

export class NegateBack implements BackwardOp {
  constructor(public input: VariableI | Tensor) {}

  backward(grad: Tensor): void {
    if ((this.input as VariableI).backward !== undefined) {
      (this.input as VariableI).backward(grad.negate());
    }
  }
}
