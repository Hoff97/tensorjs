import {Tensor} from '../../../library';
import {BackwardOp, VariableI} from '../../types';

export class MultiplyScalarBack implements BackwardOp {
  constructor(public input: VariableI, public scalar: number) {}

  backward(grad: Tensor): void {
    const gradIn = grad.multiplyScalar(this.scalar);
    const needed = this.input.backward(gradIn);
    if (!needed) {
      gradIn.delete();
    }
  }

  delete(): void {
    if (!this.input.isLeaf()) {
      this.input.delete();
    }
  }
}
