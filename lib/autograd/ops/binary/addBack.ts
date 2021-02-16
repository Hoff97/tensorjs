import {Tensor} from '../../../library';
import {BackwardOp, VariableI} from '../../types';

export class AddBack implements BackwardOp {
  constructor(
    public a: VariableI,
    public b: VariableI,
    public shape: readonly number[],
    public alpha: number,
    public beta: number
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
        if (this.alpha === 1) {
          gradA = grad.reshape(shapeA);
        } else {
          gradA = grad.multiplyScalar(this.alpha).reshape(shapeA, false);
        }
      } else {
        if (this.alpha === 1) {
          gradA = grad.sum(sumADims).reshape(shapeA, false);
        } else {
          const summed = grad.sum(sumADims);
          const scaled = summed.multiplyScalar(this.alpha);
          summed.delete();
          gradA = scaled.reshape(shapeA, false);
        }
      }
      const needed = this.a.backward(gradA);
      if (!needed) {
        gradA.delete();
      }
    }

    if (!this.b.noGrad) {
      let gradB: Tensor;
      if (sumBDims.length === 0) {
        if (this.beta === 1) {
          gradB = grad.reshape(shapeB);
        } else {
          gradB = grad.multiplyScalar(this.beta).reshape(shapeB, false);
        }
      } else {
        if (this.beta === 1) {
          gradB = grad.sum(sumBDims).reshape(shapeB, false);
        } else {
          const summed = grad.sum(sumBDims);
          const scaled = summed.multiplyScalar(this.beta);
          summed.delete();
          gradB = scaled.reshape(shapeB, false);
        }
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