import {CPUTensor} from '../../../tensor/cpu/tensor';
import {SparseTensor} from '../../../tensor/sparse/tensor';
import {WASMTensor} from '../../../tensor/wasm/tensor';
import {DType} from '../../../types';
import {reshapeCPU} from './cpu';
import {reshapeWasm} from './wasm';

export function reshape<DTpe extends DType>(
  tensor: SparseTensor<DTpe>,
  shape: readonly number[]
): SparseTensor<DTpe> {
  if (tensor.values instanceof CPUTensor) {
    return reshapeCPU(
      tensor,
      tensor.values,
      tensor.indices as CPUTensor<'uint32'>,
      shape
    );
  } else if (tensor.values instanceof WASMTensor) {
    return reshapeWasm(
      tensor,
      tensor.values,
      tensor.indices as WASMTensor<'uint32'>,
      shape
    );
  }

  throw new Error(
    'Reshape of sparse tensors on WASM/WebGL backend is currently not supported'
  );
}
