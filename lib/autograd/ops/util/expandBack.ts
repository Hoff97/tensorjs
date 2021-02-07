import {Tensor} from '../../../library';
import {BackwardOp, VariableI} from '../../types';

export class ExpandBack implements BackwardOp {
  constructor(public a: VariableI, public shape: readonly number[]) {}

  backward(grad: Tensor): void {
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    const [_shape, goal, resultShape] = this.a.value.alignShapes(
      this.a.getShape(),
      this.shape
    );

    const sumDims = [];
    for (let i = 0; i < _shape.length; i++) {
      if (_shape[i] < goal[i]) {
        sumDims.push(i);
      }
    }

    const gradA = grad.sum(sumDims).reshape(this.a.getShape()));
    const needed = this.a.backward(gradA);
    if (!needed) {
      gradA.delete();
    }
  }

  delete(): void {
    if (!this.a.isLeaf()) {
      this.a.delete();
    }
  }
}
