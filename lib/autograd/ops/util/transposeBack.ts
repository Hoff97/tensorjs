import Tensor, {DType} from '../../../types';
import {BackwardOp, VariableI} from '../../types';

export class TransposeBack<DTpe extends DType> implements BackwardOp<DTpe> {
  constructor(public a: VariableI<DTpe>, public permutation: number[]) {}

  backward(grad: Tensor<DTpe>): void {
    const inversePerm = new Array(this.permutation.length);
    for (let i = 0; i < this.permutation.length; i++) {
      inversePerm[this.permutation[i]] = i;
    }

    const gradA = grad.transpose(inversePerm);
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
