import {Tensor} from '../../../library';
import {BackwardOp, VariableI} from '../../types';

export class MultiplyScalarBack implements BackwardOp {
  constructor(public input: VariableI, public scalar: number) {}

  backward(grad: Tensor): void {
    this.input.backward(grad.multiplyScalar(this.scalar));
  }

  delete(): void {
    if (!this.input.isLeaf()) {
      this.input.delete();
    }
  }
}
