import {Tensor} from '../../../library';
import {BackwardOp, VariableI} from '../../types';

export class ConcatBack implements BackwardOp {
  constructor(
    public a: VariableI | Tensor,
    public b: VariableI | Tensor,
    public axis: number
  ) {}

  backward(grad: Tensor): void {
    if ((this.a as VariableI).backward !== undefined) {
      (this.a as VariableI).backward(
        grad.slice([0], [this.a.getShape()[this.axis]], [this.axis])
      );
    }

    if ((this.b as VariableI).backward !== undefined) {
      (this.b as VariableI).backward(
        grad.slice(
          [this.a.getShape()[this.axis]],
          [grad.getShape()[this.axis]],
          [this.axis]
        )
      );
    }
  }
}
