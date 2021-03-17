import {CPUTensor} from '../../../../tensor/cpu/tensor';
import {SparseTensor} from '../../../../tensor/sparse/tensor';
import {WASMTensor} from '../../../../tensor/wasm/tensor';
import Tensor, {DType} from '../../../../types';
import {subtractDenseCPU, subtractSparseCPU} from './cpu';
import {subtractDenseWASM} from './wasm';

export function subtract<DTpe extends DType>(
  a: SparseTensor<DTpe>,
  b: Tensor<DTpe>,
  resultShape: readonly number[],
  alpha: number,
  beta: number
): SparseTensor<DTpe> {
  if (b instanceof SparseTensor) {
    return subtractSparse(a, b, resultShape, alpha, beta);
  } else {
    return subtractDense(a, b, resultShape, alpha, beta);
  }
}

function subtractSparse<DTpe extends DType>(
  a: SparseTensor<DTpe>,
  b: SparseTensor<DTpe>,
  resultShape: readonly number[],
  alpha: number,
  beta: number
): SparseTensor<DTpe> {
  if (a.nnz !== b.nnz) {
    throw new Error(
      'Subtraction with two sparse tensors expects the same sparsity pattern, and thus the same number of nonzero entries in both tensors'
    );
  } else if (a.denseDims !== b.denseDims) {
    throw new Error(
      'Subtraction with two sparse tensors expects the same number of sparse and dense dimensions in both tensors'
    );
  }
  if (a.values instanceof CPUTensor) {
    return subtractSparseCPU(a, b, resultShape, alpha, beta);
  }
  throw new Error(
    'Sparse-sparse matrix subtraction not supported on WASM/WebGL backend'
  );
}

function subtractDense<DTpe extends DType>(
  a: SparseTensor<DTpe>,
  b: Tensor<DTpe>,
  resultShape: readonly number[],
  alpha: number,
  beta: number
): SparseTensor<DTpe> {
  if (b instanceof CPUTensor) {
    return subtractDenseCPU(a, b, resultShape, alpha, beta);
  } else if (b instanceof WASMTensor) {
    return subtractDenseWASM(a, b, resultShape, alpha, beta);
  }
  throw new Error(
    'Sparse-dense matrix addition not supported on WASM/WebGL backend'
  );
}
