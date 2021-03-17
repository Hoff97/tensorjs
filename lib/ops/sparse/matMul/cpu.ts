import {CPUTensor} from '../../../tensor/cpu/tensor';
import {SparseTensor} from '../../../tensor/sparse/tensor';
import {DType} from '../../../types';

/**
 * Calculates the sparse-dense matrix product, assuming that a
 * has zero dense dimensions.
 *
 * The result is a dense CPU tensor
 */
export function sparseDenseMatMulCPU<DTpe extends DType>(
  a: SparseTensor<DTpe>,
  b: CPUTensor<DTpe>
): CPUTensor<DTpe> {
  const M = a.shape[0];
  const O = b.shape[1];
  const result = new CPUTensor([M, O], undefined, b.dtype);

  const indices = a.indices as CPUTensor<'uint32'>;
  const values = a.values as CPUTensor<DTpe>;

  for (let i = 0; i < a.nnz; i++) {
    const m = indices.get(i * 2);
    const n = indices.get(i * 2 + 1);

    const v = values.get(i);

    for (let o = 0; o < O; o++) {
      result.set(m * O + o, result.get(m * O + o) + v * b.get(n * O + o));
    }
  }

  return result;
}
