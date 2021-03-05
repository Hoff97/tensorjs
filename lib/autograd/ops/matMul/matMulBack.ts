import {DType, Tensor} from '../../../library';
import {BackwardOp, VariableI} from '../../types';

export class MatMulBack<DTpe extends DType> implements BackwardOp<DTpe> {
  constructor(public a: VariableI<DTpe>, public b: VariableI<DTpe>) {}

  backward(grad: Tensor<DTpe>): void {
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
