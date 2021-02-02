import {Tensor} from '../../../library';
import {BackwardOp, VariableI} from '../../types';

export class MultiplyBack implements BackwardOp {
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

    if (sumADims.length === 0) {
      this.a.backward(grad.multiply(this.b.value).reshape(shapeA, false));
    } else {
      const mult = grad.multiply(this.b.value);
      const summed = mult.sum(sumADims);
      mult.delete();
      this.a.backward(summed.reshape(shapeA, false));
    }

    if (sumBDims.length === 0) {
      this.b.backward(grad.multiply(this.a.value).reshape(shapeB, false));
    } else {
      const mult = grad.multiply(this.a.value);
      const summed = mult.sum(sumBDims);
      mult.delete();
      this.b.backward(summed.reshape(shapeB, false));
    }
  }
}
