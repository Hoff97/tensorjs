import {Tensor} from '../../../library';
import {BackwardOp, VariableI} from '../../types';

export class SubtractBack implements BackwardOp {
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
      this.a.backward(grad.reshape(shapeA, false));
    } else {
      this.a.backward(grad.sum(sumADims).reshape(shapeA, false));
    }

    if (sumBDims.length === 0) {
      this.b.backward(grad.negate().reshape(shapeB, false));
    } else {
      const summed = grad.sum(sumBDims);
      const negated = summed.negate();
      summed.delete();
      this.b.backward(negated.reshape(shapeB, false));
    }
  }
}