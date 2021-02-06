import {Tensor} from '../../../library';
import {BackwardOp, VariableI} from '../../types';

export class SumSquareBack implements BackwardOp {
  constructor(
    public input: VariableI,
    public sumDims: number[],
    public keepDims: boolean
  ) {}

  backward(grad: Tensor): void {
    const inShape = this.input.value.getShape();
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
    const mult1 = expanded.multiply(this.input.value);
    expanded.delete();
    const gradIn = mult1.multiply(this.input.value.singleConstant(2));
    mult1.delete();

    this.input.backward(gradIn);
  }

  delete(): void {
    if (!this.input.isLeaf()) {
      this.input.delete();
    }
  }
}
