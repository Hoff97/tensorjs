import {DType, Tensor} from '../../../../library';
import {CPUTensor} from '../../../../tensor/cpu/tensor';
import {GPUTensor} from '../../../../tensor/gpu/tensor';
import {WASMTensor} from '../../../../tensor/wasm/tensor';
import {BackwardOp, VariableI} from '../../../types';
import {minBackCPU} from './cpu';
import {minBackGPU} from './gpu';
import {minBackWASM} from './wasm';

export class MinBack<DTpe extends DType> implements BackwardOp<DTpe> {
  constructor(public input: VariableI<DTpe>, public axes: number[]) {}

  backward(grad: Tensor<DTpe>): void {
    const gradIn = minBack(this.input.value, grad, this.axes);
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

export function minBack<DTpe extends DType>(
  value: Tensor<DTpe>,
  gradient: Tensor<DTpe>,
  axes: number[]
): Tensor<DTpe> {
  if (value instanceof CPUTensor) {
    return minBackCPU(value, gradient as CPUTensor<DTpe>, axes);
  } else if (value instanceof WASMTensor) {
    return minBackWASM(value, gradient as any, axes);
  } else if (value instanceof GPUTensor) {
    return minBackGPU(value, gradient as any, axes);
  }
  throw new Error('Backward pass of min not implemented for tensor:' + value);
}
