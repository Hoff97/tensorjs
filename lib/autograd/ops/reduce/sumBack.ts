import {Tensor} from '../../../library';
import {BackwardOp, VariableI} from '../../types';

export class SumBack implements BackwardOp {
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

    this.input.backward(grad.expand(inShape));
  }

  delete(): void {
    if (!this.input.isLeaf()) {
      this.input.delete();
    }
  }
}
