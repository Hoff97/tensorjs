import {Tensor} from '../../../library';
import {BackwardOp, VariableI} from '../../types';

export class AveragePoolBack implements BackwardOp {
  constructor(
    public x: VariableI,
    kernelShape: number[],
    pads: number[],
    strides: number[],
    includePad: boolean
  ) {}

  backward(grad: Tensor): void {
    throw new Error('Backward pass not implemented for average pool');
  }

  delete(): void {
    if (!this.x.isLeaf()) {
      this.x.delete();
    }
  }
}
