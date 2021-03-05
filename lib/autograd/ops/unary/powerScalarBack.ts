import {DType, Tensor} from '../../../library';
import {BackwardOp, VariableI} from '../../types';

export class PowerScalarBack<DTpe extends DType> implements BackwardOp<DTpe> {
  constructor(
    public input: VariableI<DTpe>,
    public power: number,
    public factor: number
  ) {}

  backward(grad: Tensor<DTpe>): void {
    const pow = this.input.value.powerScalar(
      this.power - 1,
      this.factor * this.power
    );
    const gradIn = grad.multiply(pow);
    pow.delete();
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
