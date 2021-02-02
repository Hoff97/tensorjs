import {Tensor} from '../../../library';
import {BackwardOp, VariableI} from '../../types';

export class AbsBack implements BackwardOp {
  constructor(public input: VariableI | Tensor) {}

  backward(grad: Tensor): void {
    if (
      (this.input as VariableI).backward !== undefined &&
      (this.input as VariableI).value !== undefined
    ) {
      const sign = (this.input as VariableI).value.sign();
      const gradAbs = grad.multiply(sign);
      sign.delete();
      (this.input as VariableI).backward(gradAbs);
    }
  }
}
