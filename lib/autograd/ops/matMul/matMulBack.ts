import {Tensor} from '../../../library';
import {BackwardOp, VariableI} from '../../types';

export class MatMulBack implements BackwardOp {
  constructor(public a: VariableI, public b: VariableI) {}

  backward(grad: Tensor): void {
    if (!this.b.noGrad) {
      const gradB = this.a.value.gemm(grad, true, false);
      const needed = this.b.backward(gradB);
      if (!needed) {
        gradB.delete();
      }
    }

    if (!this.a.noGrad) {
      const gradA = grad.gemm(this.b.value, false, true);
      const needed = this.a.backward(gradA);
      if (!needed) {
        gradA.delete();
      }
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
