import {DType, Tensor} from '../../../../library';
import {CPUTensor} from '../../../../tensor/cpu/tensor';
import {GPUTensor} from '../../../../tensor/gpu/tensor';
import {WASMTensor} from '../../../../tensor/wasm/tensor';
import {BackwardOp, VariableI} from '../../../types';
import {maxBackCPU} from './cpu';
import {maxBackGPU} from './gpu';
import {maxBackWASM} from './wasm';

export class MaxBack<DTpe extends DType> implements BackwardOp<DTpe> {
  constructor(public input: VariableI<DTpe>, public axes: number[]) {}

  backward(grad: Tensor<DTpe>): void {
    const gradIn = maxBack(this.input.value, grad, this.axes);
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
  gradient: Tensor<DTpe>,
  axes: number[]
): Tensor<DTpe> {
  if (value instanceof CPUTensor) {
    return maxBackCPU(value, gradient as CPUTensor<DTpe>, axes);
  } else if (value instanceof WASMTensor) {
    return maxBackWASM(value, gradient as any, axes);
  } else if (value instanceof GPUTensor) {
    return maxBackGPU(value, gradient as any, axes);
  }
  throw new Error('Backward pass of max not implemented tensor:' + value);
}
