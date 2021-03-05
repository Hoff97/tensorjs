import {DType, Tensor} from '../../../library';
import {BackwardOp, VariableI} from '../../types';

export class AveragePoolBack<DTpe extends DType> implements BackwardOp<DTpe> {
  constructor(
    public x: VariableI<DTpe>,
    kernelShape: number[],
    pads: number[],
    strides: number[],
    includePad: boolean
  ) {}

  backward(grad: Tensor<DTpe>): void {
    throw new Error('Backward pass not implemented for average pool');
  }

  delete(): void {
    if (!this.x.isLeaf()) {
      this.x.delete();
    }
  }
}
