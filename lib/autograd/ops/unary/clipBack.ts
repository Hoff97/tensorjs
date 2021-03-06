import {DType, Tensor} from '../../../library';
import {BackwardOp, VariableI} from '../../types';

export class ClipBack<DTpe extends DType> implements BackwardOp<DTpe> {
  constructor(
    public input: VariableI<DTpe>,
    public min?: number,
    public max?: number
  ) {}

  backward(grad: Tensor<DTpe>): void {
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
