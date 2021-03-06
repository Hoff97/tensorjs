import {DType, Tensor} from '../../../library';
import {BackwardOp, VariableI} from '../../types';

export class ConcatBack<DTpe extends DType> implements BackwardOp<DTpe> {
  constructor(
    public a: VariableI<DTpe>,
    public b: VariableI<DTpe>,
    public axis: number
  ) {}

  backward(grad: Tensor<DTpe>): void {
    let axis = this.axis;
    if (axis < 0) {
      axis += this.a.getShape().length;
    }

    if (!this.a.noGrad) {
      const gradA = grad.slice([0], [this.a.getShape()[axis]], [axis]);
      const needed = this.a.backward(gradA);
      if (!needed) {
        gradA.delete();
      }
    }

    if (!this.b.noGrad) {
      const gradB = grad.slice(
        [this.a.getShape()[axis]],
        [grad.getShape()[axis]],
        [axis]
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
