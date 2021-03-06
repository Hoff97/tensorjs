import Tensor, {DType} from '../../../types';
import {BackwardOp, VariableI} from '../../types';

export class SliceBack<DTpe extends DType> implements BackwardOp<DTpe> {
  constructor(
    public a: VariableI<DTpe>,
    public starts: number[],
    public ends: number[],
    public axes: number[],
    public steps: number[]
  ) {
    if (steps.find(x => x !== 1) !== undefined) {
      throw new Error('Slice backward pass only supports step size of 1');
    }
  }

  backward(grad: Tensor<DTpe>): void {
    if (!this.a.noGrad) {
      const shapeA = this.a.getShape();
      const rank = shapeA.length;

      const pads = new Array(rank * 2).fill(0);
      for (let i = 0; i < this.axes.length; i++) {
        pads[this.axes[i]] = this.starts[i];
        pads[rank + this.axes[i]] = shapeA[this.axes[i]] - this.ends[i];
      }

      const gradA = grad.pad(pads, 'constant', 0);
      const needed = this.a.backward(gradA);
      if (!needed) {
        gradA.delete();
      }
    }
  }

  delete(): void {
    if (!this.a.isLeaf()) {
      this.a.delete();
    }
  }
}
