import {CPUTensor} from '../../../tensor/cpu/tensor';
import {SparseTensor} from '../../../tensor/sparse/tensor';
import {WASMTensor} from '../../../tensor/wasm/tensor';
import Tensor, {DType} from '../../../types';
import {sparseDenseMatMulCPU} from './cpu';
import {sparseDenseMatMulWASM} from './wasm';

export function matMul<DTpe extends DType>(
  a: SparseTensor<DTpe>,
  b: Tensor<DTpe>
) {
  if (b instanceof SparseTensor) {
    throw new Error('Sparse-sparse matrix multiplication not yet implemented');
  } else {
    return sparseDenseMatMul(a, b);
  }
}

function sparseDenseMatMul<DTpe extends DType>(
  a: SparseTensor<DTpe>,
  b: Tensor<DTpe>
): Tensor<DTpe> {
  if (a.denseDims === 1) {
    return new SparseTensor(
      a.values.matMul(b),
      a.indices.copy(),
      [a.shape[0], b.getShape()[1]],
      1
    );
  }

  if (b instanceof CPUTensor) {
    return sparseDenseMatMulCPU(a, b);
  } else if (b instanceof WASMTensor) {
    return sparseDenseMatMulWASM(a, b);
  }
  throw new Error(
    'Sparse-dense matrix multiplication not yet supported on WebGL'
  );
}
