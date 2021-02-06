import {Tensor} from '../../../library';
import {BackwardOp, VariableI} from '../../types';

export class TransposeBack implements BackwardOp {
  constructor(public a: VariableI, public permutation: number[]) {}

  backward(grad: Tensor): void {
    const inversePerm = new Array(this.permutation.length);
    for (let i = 0; i < this.permutation.length; i++) {
      inversePerm[this.permutation[i]] = i;
    }

    this.a.backward(grad.transpose(inversePerm));
  }

  delete(): void {
    if (!this.a.isLeaf()) {
      this.a.delete();
    }
  }
}
