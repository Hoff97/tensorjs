import {Tensor} from '../../../library';
import {BackwardOp, VariableI} from '../../types';

export class MatMulBack implements BackwardOp {
  constructor(public a: VariableI, public b: VariableI) {}

  backward(grad: Tensor): void {
    if (!this.b.noGrad) {
      const gradB = this.a.value.gemm(grad, true, false);
      this.b.backward(gradB);
    }

    if (!this.a.noGrad) {
      const gradA = grad.gemm(this.b.value, false, true);
      this.a.backward(gradA);
    }
  }

  delete(): void {
    if (!this.a.isLeaf()) {
      this.a.delete();
    }

    if (!this.b.isLeaf()) {
      this.b.delete();
    }
  }
}
