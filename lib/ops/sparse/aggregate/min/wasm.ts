import {SparseTensor} from '../../../../tensor/sparse/tensor';
import {DTypeWasm, WASMTensor} from '../../../../tensor/wasm/tensor';

export function minSparseWASM<DTpe extends DTypeWasm>(
  tensor: SparseTensor<DTpe>,
  axes: number[],
  keepDims: boolean
): WASMTensor<DTpe> {
  return new WASMTensor(
    (tensor.values as WASMTensor<DTpe>).wasmTensor.min_sparse(
      new Uint32Array(tensor.getShape()),
      (tensor.indices as WASMTensor<'uint32'>).wasmTensor,
      new Uint32Array(axes),
      keepDims
    ) as any,
    undefined,
    tensor.dtype
  );
}
