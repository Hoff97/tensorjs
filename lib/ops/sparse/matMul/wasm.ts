import {SparseTensor} from '../../../tensor/sparse/tensor';
import {DTypeWasm, WASMTensor} from '../../../tensor/wasm/tensor';

/**
 * Calculates the sparse-dense matrix product, assuming that a
 * has zero dense dimensions.
 *
 * The result is a dense CPU tensor
 */
export function sparseDenseMatMulWASM<DTpe extends DTypeWasm>(
  a: SparseTensor<DTpe>,
  b: WASMTensor<DTpe>
): WASMTensor<DTpe> {
  return new WASMTensor(
    (a.values as WASMTensor<DTpe>).wasmTensor.matmul_sparse_dense(
      (a.indices as WASMTensor<'uint32'>).wasmTensor,
      b.wasmTensor as any,
      a.shape[0]
    ) as any,
    undefined,
    a.dtype
  );
}
