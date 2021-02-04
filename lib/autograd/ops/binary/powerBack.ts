import {Tensor} from '../../../library';
import {BackwardOp, VariableI} from '../../types';

export class PowerBack implements BackwardOp {
  constructor(
    public a: VariableI,
    public b: VariableI,
    public powerResult: Tensor,
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
      const multiplied = this.powerResult.multiply(this.b.value);
      const divided = multiplied.divide(this.a.value);
      multiplied.delete();
      const gradPowA = grad.multiply(divided);
      divided.delete();
      this.a.backward(gradPowA.reshape(shapeA, false));
    } else {
      const multiplied = this.powerResult.multiply(this.b.value);
      const divided = multiplied.divide(this.a.value);
      multiplied.delete();
      const gradPowA = grad.multiply(divided);
      divided.delete();
      const summed = gradPowA.sum(sumADims);
      gradPowA.delete();
      this.a.backward(summed.reshape(shapeA, false));
    }

    if (sumBDims.length === 0) {
      const lnA = this.a.value.log();
      const mult = this.powerResult.multiply(lnA);
      lnA.delete();
      const gradB = grad.multiply(mult);
      mult.delete();

      this.b.backward(gradB.reshape(shapeB, false));
    } else {
      const lnA = this.a.value.log();
      const mult = this.powerResult.multiply(lnA);
      lnA.delete();
      const gradB = grad.multiply(mult);
      mult.delete();
      const summed = gradB.sum(sumBDims);
      gradB.delete();
      this.b.backward(summed.reshape(shapeB, false));
    }
  }
}
