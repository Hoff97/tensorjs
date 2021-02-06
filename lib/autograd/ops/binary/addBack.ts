import {Tensor} from '../../../library';
import {BackwardOp, VariableI} from '../../types';

export class AddBack implements BackwardOp {
  constructor(
    public a: VariableI,
    public b: VariableI,
    public shape: readonly number[]
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
        this.a.backward(grad.reshape(shapeA, false));
      } else {
        this.a.backward(grad.sum(sumADims).reshape(shapeA, false));
      }
    }

    if (!this.b.noGrad) {
      if (sumBDims.length === 0) {
        this.b.backward(grad.reshape(shapeB, false));
      } else {
        this.b.backward(grad.sum(sumBDims).reshape(shapeB, false));
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
