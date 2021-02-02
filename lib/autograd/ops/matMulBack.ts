import {Tensor} from '../../library';
import {BackwardOp, VariableI} from '../types';

export class MatMulBack implements BackwardOp {
  constructor(public a: VariableI | Tensor, public b: VariableI | Tensor) {}

  backward(grad: Tensor): void {
    if (
      (this.a as VariableI).value !== undefined &&
      (this.b as VariableI).backward !== undefined
    ) {
      const gradB = (this.a as VariableI).value.gemm(grad, true, false);
      (this.b as VariableI).backward(gradB);
    }

    if (
      (this.b as VariableI).value !== undefined &&
      (this.a as VariableI).backward !== undefined
    ) {
      const gradB = grad.gemm((this.b as VariableI).value, false, true);
      (this.a as VariableI).backward(gradB);
    }
  }
}
