import {SparseTensor} from '../../../../tensor/sparse/tensor';
import {DTypeWasm, WASMTensor} from '../../../../tensor/wasm/tensor';

export function multiplyDenseWASM<DTpe extends DTypeWasm>(
  a: SparseTensor<DTpe>,
  b: WASMTensor<DTpe>,
  resultShape: readonly number[],
  alpha: number
): SparseTensor<DTpe> {
  const vals = new WASMTensor(
    (a.values as WASMTensor<DTpe>).wasmTensor.multiply_sparse_dense(
      (a.indices as WASMTensor<'uint32'>).wasmTensor,
      b.wasmTensor as any,
      new Uint32Array(resultShape),
      alpha
    ) as any,
    undefined,
    a.dtype
  );

  return new SparseTensor(vals, a.indices.copy(), resultShape, a.denseDims);
}
