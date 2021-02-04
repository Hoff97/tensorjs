import {Tensor} from '../../../library';
import {BackwardOp, VariableI} from '../../types';

export class DivideBack implements BackwardOp {
  constructor(
    public a: VariableI,
    public b: VariableI,
    public divResult: Tensor,
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
      this.a.backward(grad.divide(this.b.value).reshape(shapeA, false));
    } else {
      const mult = grad.divide(this.b.value);
      const summed = mult.sum(sumADims);
      mult.delete();
      this.a.backward(summed.reshape(shapeA, false));
    }

    if (sumBDims.length === 0) {
      const multiplied = grad.multiply(this.divResult);
      const divided = multiplied.divide(this.b.value);
      multiplied.delete();
      const negated = divided.negate();
      divided.delete();

      this.b.backward(negated.reshape(shapeB, false));
    } else {
      const multiplied = grad.multiply(this.divResult);
      const divided = multiplied.divide(this.b.value);
      multiplied.delete();
      const negated = divided.negate();
      divided.delete();
      const summed = negated.sum(sumBDims);
      negated.delete();
      this.b.backward(summed.reshape(shapeB, false));
    }
  }
}
