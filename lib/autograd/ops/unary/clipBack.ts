import {Tensor} from '../../../library';
import {BackwardOp, VariableI} from '../../types';

export class ClipBack implements BackwardOp {
  constructor(
    public input: VariableI,
    public min?: number,
    public max?: number
  ) {}

  backward(grad: Tensor): void {
    this.input.backward(
      this.input.value.clipBackward(grad, this.min, this.max)
    );
  }

  delete(): void {
    if (!this.input.isLeaf()) {
      this.input.delete();
    }
  }
}
