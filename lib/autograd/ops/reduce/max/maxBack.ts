import {DType, Tensor} from '../../../../library';
import {CPUTensor} from '../../../../tensor/cpu/tensor';
import {BackwardOp, VariableI} from '../../../types';

export class MaxBack<DTpe extends DType> implements BackwardOp<DTpe> {
  constructor(
    public input: VariableI<DTpe>,
    public sumDims: number[],
    public keepDims: boolean
  ) {}

  backward(grad: Tensor<DTpe>): void {
    const gradIn = maxBack(this.input.value, grad);
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

export function maxBack<DTpe extends DType>(
  value: Tensor<DTpe>,
  gradient: Tensor<DTpe>
): Tensor<DTpe> {
  if (value instanceof CPUTensor) {
  }
  throw new Error('Backward pass of max not implemented for WASM/WebGL yet');
}
