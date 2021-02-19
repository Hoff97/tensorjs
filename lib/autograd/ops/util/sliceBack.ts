import {Tensor} from '../../../library';
import {BackwardOp, VariableI} from '../../types';

export class SliceBack implements BackwardOp {
  constructor(
    public a: VariableI,
    public starts: number[],
    public ends: number[],
    public axes: number[],
    public steps: number[]
  ) {
    if (steps.find(x => x !== 1) !== undefined) {
      throw new Error('Slice backward pass only supports step size of 1');
    }
  }

  backward(grad: Tensor): void {
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
