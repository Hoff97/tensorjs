import {Tensor} from '../../../library';
import {BackwardOp, VariableI} from '../../types';

export class SubtractBack implements BackwardOp {
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
      if (sumADims.length === 0) {
        if (this.alpha === 1) {
          this.a.backward(grad.reshape(shapeA));
        } else {
          this.a.backward(
            grad.multiplyScalar(this.alpha).reshape(shapeA, false)
          );
        }
      } else {
        if (this.alpha === 1) {
          this.a.backward(grad.sum(sumADims).reshape(shapeA, false));
        } else {
          const summed = grad.sum(sumADims);
          const scaled = summed.multiplyScalar(this.alpha);
          summed.delete();
          this.a.backward(scaled.reshape(shapeA, false));
        }
      }
    }

    if (!this.b.noGrad) {
      if (sumBDims.length === 0) {
        this.b.backward(grad.multiplyScalar(-this.beta).reshape(shapeB, false));
      } else {
        const summed = grad.sum(sumBDims);
        const scaled = summed.multiplyScalar(-this.beta);
        summed.delete();
        this.b.backward(scaled.reshape(shapeB, false));
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
