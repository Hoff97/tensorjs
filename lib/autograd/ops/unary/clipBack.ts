import {Tensor} from '../../../library';
import {BackwardOp, VariableI} from '../../types';

export class ClipBack implements BackwardOp {
  constructor(
    public input: VariableI,
    public min?: number,
    public max?: number
  ) {}

  backward(grad: Tensor): void {
    const gradIn = this.input.value.clipBackward(grad, this.min, this.max);
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
