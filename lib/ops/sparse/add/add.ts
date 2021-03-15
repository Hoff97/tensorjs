import {CPUTensor} from '../../../tensor/cpu/tensor';
import {SparseTensor} from '../../../tensor/sparse/tensor';
import Tensor, {DType} from '../../../types';
import {addDenseCPU, addSparseCPU} from './cpu';

export function add<DTpe extends DType>(
  a: SparseTensor<DTpe>,
  b: Tensor<DTpe>,
  resultShape: readonly number[],
  alpha: number,
  beta: number
): SparseTensor<DTpe> {
  if (b instanceof SparseTensor) {
    return addSparse(a, b, resultShape, alpha, beta);
  } else {
    return addDense(a, b, resultShape, alpha, beta);
  }
}

function addSparse<DTpe extends DType>(
  a: SparseTensor<DTpe>,
  b: SparseTensor<DTpe>,
  resultShape: readonly number[],
  alpha: number,
  beta: number
): SparseTensor<DTpe> {
  if (a.nnz !== b.nnz) {
    throw new Error(
      'Addition with two sparse tensors expects the same sparsity pattern, and thus the same number of nonzero entries in both tensors'
    );
  } else if (a.denseDims !== b.denseDims) {
    throw new Error(
      'Addition with two sparse tensors expects the same number of sparse and dense dimensions in both tensors'
    );
  }
  if (a.values instanceof CPUTensor) {
    return addSparseCPU(a, b, resultShape, alpha, beta);
  }
  throw new Error(
    'Sparse-sparse matrix addition not supported on WASM/WebGL backend'
  );
}

function addDense<DTpe extends DType>(
  a: SparseTensor<DTpe>,
  b: Tensor<DTpe>,
  resultShape: readonly number[],
  alpha: number,
  beta: number
): SparseTensor<DTpe> {
  if (b instanceof CPUTensor) {
    return addDenseCPU(a, b, resultShape, alpha, beta);
  }
  throw new Error(
    'Sparse-dense matrix addition not supported on WASM/WebGL backend'
  );
}
