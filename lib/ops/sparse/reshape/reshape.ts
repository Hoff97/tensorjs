import {CPUTensor} from '../../../tensor/cpu/tensor';
import {GPUTensor} from '../../../tensor/gpu/tensor';
import {SparseTensor} from '../../../tensor/sparse/tensor';
import {WASMTensor} from '../../../tensor/wasm/tensor';
import {DType} from '../../../types';
import {reshapeCPU} from './cpu';
import {reshapeGPU} from './gpu';
import {reshapeWasm} from './wasm';

export function reshape<DTpe extends DType>(
  tensor: SparseTensor<DTpe>,
  shape: readonly number[],
  copy: boolean
): SparseTensor<DTpe> {
  if (tensor.values instanceof CPUTensor) {
    return reshapeCPU(
      tensor,
      tensor.values,
      tensor.indices as CPUTensor<'uint32'>,
      shape,
      copy
    );
  } else if (tensor.values instanceof WASMTensor) {
    return reshapeWasm(
      tensor,
      tensor.values,
      tensor.indices as WASMTensor<'uint32'>,
      shape,
      copy
    );
  } else {
    return reshapeGPU(
      tensor as any,
      tensor.values as any,
      tensor.indices as any,
      shape,
      copy
    ) as any;
  }
}
