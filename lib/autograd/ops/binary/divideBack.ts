import {Tensor} from '../../../library';
import {BackwardOp, VariableI} from '../../types';

export class DivideBack implements BackwardOp {
  constructor(
    public a: VariableI,
    public b: VariableI,
    public divResult: Tensor,
    public shape: readonly number[],
    public alpha: number
  ) {}

  backward(grad: Tensor): void {
    const shapeA = this.a.getShape();
    const shapeB = this.b.getShape();

    const sumADims = [];
    const sumBDims = [];
    for (let i = 0; i < this.shape.length; i++) {
      if (shapeA[i] < this.shape[i]) {
        sumADims.push(i);
      }

      if (shapeB[i] < this.shape[i]) {
        sumBDims.push(i);
      }
    }

    if (!this.a.noGrad) {
      let gradA: Tensor;
      if (sumADims.length === 0) {
        gradA = grad.divide(this.b.value, this.alpha).reshape(shapeA, false);
      } else {
        const mult = grad.divide(this.b.value, this.alpha);
        const summed = mult.sum(sumADims);
        mult.delete();
        gradA = summed.reshape(shapeA, false);
      }
      const needed = this.a.backward(gradA);
      if (!needed) {
        gradA.delete();
      }
    }

    if (!this.b.noGrad) {
      let gradB: Tensor;
      if (sumBDims.length === 0) {
        const multiplied = grad.multiply(this.divResult);
        const divided = multiplied.divide(this.b.value, -this.alpha);
        multiplied.delete();

        gradB = divided.reshape(shapeB, false);
      } else {
        const multiplied = grad.multiply(this.divResult);
        const divided = multiplied.divide(this.b.value, -this.alpha);
        multiplied.delete();
        const summed = divided.sum(sumBDims);
        divided.delete();
        gradB = summed.reshape(shapeB, false);
      }
      const needed = this.b.backward(gradB);
      if (!needed) {
        gradB.delete();
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
