import {Tensor} from '../../../library';
import {BackwardOp, VariableI} from '../../types';

export class ClipBack implements BackwardOp {
  constructor(
    public input: VariableI | Tensor,
    public min?: number,
    public max?: number
  ) {}

  backward(grad: Tensor): void {
    if (
      (this.input as VariableI).backward !== undefined &&
      (this.input as VariableI).value !== undefined
    ) {
      (this.input as VariableI).backward(
        (this.input as VariableI).value.clipBackward(grad, this.min, this.max)
      );
    }
  }
}
