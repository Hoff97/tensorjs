import {DType, Tensor} from '../../../library';
import {BackwardOp, VariableI} from '../../types';

export class ProductBack<DTpe extends DType> implements BackwardOp<DTpe> {
  constructor(
    public input: VariableI<DTpe>,
    public product: Tensor<DTpe>,
    public sumDims: number[],
    public keepDims: boolean
  ) {}

  backward(grad: Tensor<DTpe>): void {
    const inShape = this.input.value.getShape();

    let mult = grad.multiply(this.product);
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

      mult = mult.reshape(newShape, false);
    }

    const expanded = mult.expand(inShape);
    mult.delete();
    const gradIn = expanded.divide(this.input.value);
    expanded.delete();

    const needed = this.input.backward(gradIn);
    if (!needed) {
      gradIn.delete();
    }
  }

  delete(): void {
    if (!this.input.isLeaf()) {
      this.input.delete();
    }
  }
}
