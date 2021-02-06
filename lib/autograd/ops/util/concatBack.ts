import {Tensor} from '../../../library';
import {BackwardOp, VariableI} from '../../types';

export class ConcatBack implements BackwardOp {
  constructor(public a: VariableI, public b: VariableI, public axis: number) {}

  backward(grad: Tensor): void {
    if (!this.a.noGrad) {
      this.a.backward(
        grad.slice([0], [this.a.getShape()[this.axis]], [this.axis])
      );
    }

    if (!this.b.noGrad) {
      this.b.backward(
        grad.slice(
          [this.a.getShape()[this.axis]],
          [grad.getShape()[this.axis]],
          [this.axis]
        )
      );
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
