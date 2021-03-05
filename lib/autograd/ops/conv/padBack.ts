import {DType, PadMode, Tensor} from '../../../library';
import {BackwardOp, VariableI} from '../../types';

export class PadBack<DTpe extends DType> implements BackwardOp<DTpe> {
  constructor(
    public x: VariableI<DTpe>,
    public pads: number[],
    public mode: PadMode,
    public value: number
  ) {}

  backward(grad: Tensor<DTpe>): void {
    throw new Error('Backward pass not implemented for pad');
  }

  delete(): void {
    if (!this.x.isLeaf()) {
      this.x.delete();
    }
  }
}
