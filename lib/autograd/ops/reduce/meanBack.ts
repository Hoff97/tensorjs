import {DType, Tensor} from '../../../library';
import {BackwardOp, VariableI} from '../../types';

export class MeanBack<DTpe extends DType> implements BackwardOp<DTpe> {
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

    let sumSize = 1;
    for (let i = 0; i < this.sumDims.length; i++) {
      sumSize *= inShape[this.sumDims[i]];
    }

    const multiplied = grad.multiplyScalar(1 / sumSize);
    const gradIn = multiplied.expand(inShape);
    multiplied.delete;
    const needed = this.input.backward(gradIn);
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
