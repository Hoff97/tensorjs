import {Tensor} from '../../../library';
import {BackwardOp, VariableI} from '../../types';

export class ReshapeBack implements BackwardOp {
  constructor(public a: VariableI) {}

  backward(grad: Tensor): void {
    const shapeA = this.a.getShape();

    if (!this.a.noGrad) {
      const gradA = grad.reshape(shapeA);
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
