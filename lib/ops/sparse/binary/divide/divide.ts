import {CPUTensor} from '../../../../tensor/cpu/tensor';
import {SparseTensor} from '../../../../tensor/sparse/tensor';
import {WASMTensor} from '../../../../tensor/wasm/tensor';
import Tensor, {DType} from '../../../../types';
import {divideDenseCPU, divideSparseCPU} from './cpu';
import {divideDenseWASM} from './wasm';

export function divide<DTpe extends DType>(
  a: SparseTensor<DTpe>,
  b: Tensor<DTpe>,
  resultShape: readonly number[],
  alpha: number
): SparseTensor<DTpe> {
  if (b instanceof SparseTensor) {
    return divideSparse(a, b, resultShape, alpha);
  } else {
    return divideDense(a, b, resultShape, alpha);
  }
}

function divideSparse<DTpe extends DType>(
  a: SparseTensor<DTpe>,
  b: SparseTensor<DTpe>,
  resultShape: readonly number[],
  alpha: number
): SparseTensor<DTpe> {
  if (a.nnz !== b.nnz) {
    throw new Error(
      'Element wise division with two sparse tensors expects the same sparsity pattern, and thus the same number of nonzero entries in both tensors'
    );
  } else if (a.denseDims !== b.denseDims) {
    throw new Error(
      'Element wise division with two sparse tensors expects the same number of sparse and dense dimensions in both tensors'
    );
  }
  if (a.values instanceof CPUTensor) {
    return divideSparseCPU(a, b, resultShape, alpha);
  }
  throw new Error(
    'Sparse-sparse matrix division not supported on WASM/WebGL backend'
  );
}

function divideDense<DTpe extends DType>(
  a: SparseTensor<DTpe>,
  b: Tensor<DTpe>,
  resultShape: readonly number[],
  alpha: number
): SparseTensor<DTpe> {
  if (b instanceof CPUTensor) {
    return divideDenseCPU(a, b, resultShape, alpha);
  } else if (b instanceof WASMTensor) {
    return divideDenseWASM(a, b, resultShape, alpha);
  }
  throw new Error(
    'Sparse-dense matrix element wise division not supported on WebGL backend'
  );
}
