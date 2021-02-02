import {Tensor} from '../../library';
import {BackwardOp, VariableI} from '../types';

export class MatMulBack implements BackwardOp {
  constructor(public a: VariableI, public b: VariableI) {}

  backward(grad: Tensor): void {
    const gradB = this.a.value.gemm(grad, true, false);
    this.b.backward(gradB);

    const gradA = grad.gemm(this.b.value, false, true);
    this.a.backward(gradA);
  }
}
