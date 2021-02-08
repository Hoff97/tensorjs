import {PadMode, Tensor} from '../../../library';
import {BackwardOp, VariableI} from '../../types';

export class PadBack implements BackwardOp {
  constructor(
    public x: VariableI,
    public pads: number[],
    public mode: PadMode,
    public value: number
  ) {}

  backward(grad: Tensor): void {
    throw new Error('Backward pass not implemented for pad');
  }

  delete(): void {
    if (!this.x.isLeaf()) {
      this.x.delete();
    }
  }
}
