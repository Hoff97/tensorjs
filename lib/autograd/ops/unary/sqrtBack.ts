import {Tensor} from '../../../library';
import {BackwardOp, VariableI} from '../../types';

export class SqrtBack implements BackwardOp {
  constructor(public input: VariableI, public sqrt: Tensor) {}

  backward(grad: Tensor): void {
    const doubleSqrt = this.sqrt.multiply(this.sqrt.singleConstant(2));
    const gradSqrt = grad.divide(doubleSqrt);
    doubleSqrt.delete();
    this.input.backward(gradSqrt);
  }
}
