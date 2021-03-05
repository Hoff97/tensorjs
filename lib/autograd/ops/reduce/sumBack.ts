import {DType, Tensor} from '../../../library';
import {BackwardOp, VariableI} from '../../types';

export class SumBack<DTpe extends DType> implements BackwardOp<DTpe> {
  constructor(
    public input: VariableI<DTpe>,
    public sumDims: number[],
    public keepDims: boolean
  ) {}

  backward(grad: Tensor<DTpe>): void {
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

    grad = grad.expand(inShape);
    const needed = this.input.backward(grad);
    if (!needed) {
      grad.delete();
    }
  }

  delete(): void {
    if (!this.input.isLeaf()) {
      this.input.delete();
    }
  }
}
