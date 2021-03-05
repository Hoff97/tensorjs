import {DType, Tensor} from '../../../library';
import {BackwardOp, VariableI} from '../../types';

export class AddMultiplyScalarBack<DTpe extends DType>
  implements BackwardOp<DTpe> {
  constructor(public input: VariableI<DTpe>, public scalar: number) {}

  backward(grad: Tensor<DTpe>): void {
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
