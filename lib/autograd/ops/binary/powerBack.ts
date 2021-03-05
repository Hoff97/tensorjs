import {DType, Tensor} from '../../../library';
import {BackwardOp, VariableI} from '../../types';

export class PowerBack<DTpe extends DType> implements BackwardOp<DTpe> {
  constructor(
    public a: VariableI<DTpe>,
    public b: VariableI<DTpe>,
    public powerResult: Tensor<DTpe>,
    public shape: readonly number[]
  ) {}

  backward(grad: Tensor<DTpe>): void {
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
      let gradA: Tensor<DTpe>;
      if (sumADims.length === 0) {
        const multiplied = this.powerResult.multiply(this.b.value);
        const divided = multiplied.divide(this.a.value);
        multiplied.delete();
        const gradPowA = grad.multiply(divided);
        divided.delete();
        gradA = gradPowA.reshape(shapeA, false);
      } else {
        const multiplied = this.powerResult.multiply(this.b.value);
        const divided = multiplied.divide(this.a.value);
        multiplied.delete();
        const gradPowA = grad.multiply(divided);
        divided.delete();
        const summed = gradPowA.sum(sumADims);
        gradPowA.delete();
        gradA = summed.reshape(shapeA, false);
      }
      const needed = this.a.backward(gradA);
      if (!needed) {
        gradA.delete();
      }
    }

    if (!this.b.noGrad) {
      let gradB: Tensor<DTpe>;
      if (sumBDims.length === 0) {
        const lnA = this.a.value.log();
        const mult = this.powerResult.multiply(lnA);
        lnA.delete();
        gradB = grad.multiply(mult);
        mult.delete();

        gradB = gradB.reshape(shapeB, false);
      } else {
        const lnA = this.a.value.log();
        const mult = this.powerResult.multiply(lnA);
        lnA.delete();
        const _gradB = grad.multiply(mult);
        mult.delete();
        const summed = _gradB.sum(sumBDims);
        _gradB.delete();
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
