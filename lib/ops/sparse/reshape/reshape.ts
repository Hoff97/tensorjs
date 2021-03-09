import {CPUTensor} from '../../../tensor/cpu/tensor';
import {SparseTensor} from '../../../tensor/sparse/tensor';
import {DType} from '../../../types';
import {getSize} from '../../../util/shape';
import {reshapeCPU} from './cpu';

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
  }

  throw new Error(
    'Reshape of sparse tensors on WASM/WebGL backend is currently not supported'
  );
}
