import {Tensor} from '../../../library';
import {BackwardOp, VariableI} from '../../types';

export class NegateBack implements BackwardOp {
  constructor(public input: VariableI) {}

  backward(grad: Tensor): void {
    this.input.backward(grad.negate());
  }
}
