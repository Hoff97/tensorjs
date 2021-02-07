import {Tensor} from '../../../library';
import {BackwardOp, VariableI} from '../../types';

export class RepeatBack implements BackwardOp {
  constructor(public a: VariableI, public repeats: number[]) {}

  backward(grad: Tensor): void {
    const shapeA = this.a.getShape();

    const gradNewShape = [];
    const sumAxes = [];
    for (let i = 0; i < shapeA.length; i++) {
      gradNewShape.push(this.repeats[i], shapeA[i]);
      sumAxes.push(i * 2);
    }

    const gradA = grad.reshape(gradNewShape, false).sum(sumAxes, false);
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
