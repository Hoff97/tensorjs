import {CPUTensor} from '../../../../tensor/cpu/tensor';
import {SparseTensor} from '../../../../tensor/sparse/tensor';
import Tensor, {DType} from '../../../../types';
import {multiplyDenseCPU, multiplySparseCPU} from './cpu';

export function multiply<DTpe extends DType>(
  a: SparseTensor<DTpe>,
  b: Tensor<DTpe>,
  resultShape: readonly number[],
  alpha: number
): SparseTensor<DTpe> {
  if (b instanceof SparseTensor) {
    return multiplySparse(a, b, resultShape, alpha);
  } else {
    return multiplyDense(a, b, resultShape, alpha);
  }
}

function multiplySparse<DTpe extends DType>(
  a: SparseTensor<DTpe>,
  b: SparseTensor<DTpe>,
  resultShape: readonly number[],
  alpha: number
): SparseTensor<DTpe> {
  if (a.nnz !== b.nnz) {
    throw new Error(
      'Element wise multiplication with two sparse tensors expects the same sparsity pattern, and thus the same number of nonzero entries in both tensors'
    );
  } else if (a.denseDims !== b.denseDims) {
    throw new Error(
      'Element wise multiplication with two sparse tensors expects the same number of sparse and dense dimensions in both tensors'
    );
  }
  if (a.values instanceof CPUTensor) {
    return multiplySparseCPU(a, b, resultShape, alpha);
  }
  throw new Error(
    'Sparse-sparse matrix addition not supported on WASM/WebGL backend'
  );
}

function multiplyDense<DTpe extends DType>(
  a: SparseTensor<DTpe>,
  b: Tensor<DTpe>,
  resultShape: readonly number[],
  alpha: number
): SparseTensor<DTpe> {
  if (b instanceof CPUTensor) {
    return multiplyDenseCPU(a, b, resultShape, alpha);
  }
  throw new Error(
    'Sparse-dense matrix element wise multiplication not supported on WASM/WebGL backend'
  );
}
