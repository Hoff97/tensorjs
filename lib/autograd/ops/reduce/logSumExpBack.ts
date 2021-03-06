import {DType, Tensor} from '../../../library';
import {BackwardOp, VariableI} from '../../types';

export class LogSumExpBack<DTpe extends DType> implements BackwardOp<DTpe> {
  constructor(
    public input: VariableI<DTpe>,
    public sumDims: number[],
    public keepDims: boolean
  ) {}

  backward(grad: Tensor<DTpe>): void {
    const inShape = this.input.value.getShape();

    const exp = this.input.value.exp();
    const sum = exp.sum(this.sumDims, true);
    const div = exp.divide(sum);
    exp.delete();
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

      grad = grad.reshape(newShape, false);
    }

    const expanded = grad.expand(inShape);

    const gradLogSumExp = expanded.multiply(div);
    expanded.delete();
    const needed = this.input.backward(gradLogSumExp);
    if (!needed) {
      gradLogSumExp.delete();
    }
  }

  delete(): void {
    if (!this.input.isLeaf()) {
      this.input.delete();
    }
  }
}
