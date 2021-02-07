import {Tensor} from '../../../library';
import {BackwardOp, VariableI} from '../../types';

export class ConcatBack implements BackwardOp {
  constructor(public a: VariableI, public b: VariableI, public axis: number) {}

  backward(grad: Tensor): void {
    if (!this.a.noGrad) {
      const gradA = grad.slice(
        [0],
        [this.a.getShape()[this.axis]],
        [this.axis]
      );
      const needed = this.a.backward(gradA);
      if (!needed) {
        gradA.delete();
      }
    }

    if (!this.b.noGrad) {
      const gradB = grad.slice(
        [this.a.getShape()[this.axis]],
        [grad.getShape()[this.axis]],
        [this.axis]
      );
      const needed = this.b.backward(gradB);
      if (!needed) {
        gradB.delete();
      }
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
