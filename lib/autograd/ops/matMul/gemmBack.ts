import {Tensor} from '../../../library';
import {BackwardOp, VariableI} from '../../types';

export class GemmBack implements BackwardOp {
  constructor(
    public a: VariableI,
    public b: VariableI,
    public transA: boolean,
    public transB: boolean,
    public alpha: number,
    public beta: number,
    public c?: VariableI
  ) {}

  backward(grad: Tensor): void {
    if (!this.b.noGrad) {
      let gradB: Tensor;
      if (this.transB) {
        gradB = grad.gemm(this.a.value, true, this.transA, this.alpha);
      } else {
        gradB = this.a.value.gemm(grad, !this.transA, false, this.alpha);
      }
      const needed = this.b.backward(gradB);
      if (!needed) {
        gradB.delete();
      }
    }

    if (!this.a.noGrad) {
      let gradA: Tensor;
      if (this.transA) {
        gradA = this.b.value.gemm(grad, this.transB, true, this.alpha);
      } else {
        gradA = grad.gemm(this.b.value, false, !this.transB, this.alpha);
      }
      const needed = this.a.backward(gradA);
      if (!needed) {
        gradA.delete();
      }
    }

    if (this.c !== undefined && !this.c.noGrad) {
      const gradShape = grad.getShape();
      const cShape = this.c.getShape();
      const cSumDims = [];

      for (let i = 0; i < gradShape.length; i++) {
        if (cShape[i] < gradShape[i]) {
          cSumDims.push(i);
        }
      }

      let gradC = grad.sum(cSumDims).reshape(cShape, false);
      if (this.beta !== 1) {
        const oldGradC = gradC;
        gradC = gradC.multiplyScalar(this.beta);
        oldGradC.delete();
      }

      const needed = this.c.backward(gradC);
      if (!needed) {
        gradC.delete();
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

    if (this.c !== undefined && !this.c.isLeaf()) {
      this.c.delete();
    }
  }
}
