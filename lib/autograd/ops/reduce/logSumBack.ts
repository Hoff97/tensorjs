import {DType, Tensor} from '../../../library';
import {BackwardOp, VariableI} from '../../types';

export class LogSumBack<DTpe extends DType> implements BackwardOp<DTpe> {
  constructor(
    public input: VariableI<DTpe>,
    public sumDims: number[],
    public keepDims: boolean
  ) {}

  backward(grad: Tensor<DTpe>): void {
    const inShape = this.input.value.getShape();

    const sum = this.input.value.sum(this.sumDims, this.keepDims);
    let gradLogSum = grad.divide(sum);
    sum.delete();

    if (!this.keepDims) {
      const newShape = [];
      let sumI = 0;
      for (let i = 0; i < inShape.length; i++) {
        if (sumI < this.sumDims.length && this.sumDims[sumI] === i) {
          newShape.push(1);
          sumI++;
        } else {
          newShape.push(inShape[i]);
        }
      }

      gradLogSum = gradLogSum.reshape(newShape, false);
    }

    const expanded = gradLogSum.expand(inShape);
    gradLogSum.delete();

    const needed = this.input.backward(expanded);
    if (!needed) {
      expanded.delete();
    }
  }

  delete(): void {
    if (!this.input.isLeaf()) {
      this.input.delete();
    }
  }
}
