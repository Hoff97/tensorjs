import {DType, Tensor} from '../../../library';
import {BackwardOp, VariableI} from '../../types';

export class ReshapeBack<DTpe extends DType> implements BackwardOp<DTpe> {
  constructor(public a: VariableI<DTpe>) {}

  backward(grad: Tensor<DTpe>): void {
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
