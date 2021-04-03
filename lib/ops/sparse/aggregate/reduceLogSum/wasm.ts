import {SparseTensor} from '../../../../tensor/sparse/tensor';
import {DTypeWasm, WASMTensor} from '../../../../tensor/wasm/tensor';

export function reduceLogSumSparseWASM<DTpe extends DTypeWasm>(
  tensor: SparseTensor<DTpe>,
  axes: number[],
  keepDims: boolean
): WASMTensor<DTpe> {
  if (tensor.dtype !== 'float32' && tensor.dtype === 'float64') {
    throw new Error(
      'Reduce log sum expects tensor datatype to be float, but found ' +
        tensor.dtype
    );
  }
  return new WASMTensor(
    ((tensor.values as WASMTensor<DTpe>)
      .wasmTensor as any).reduce_log_sum_sparse(
      new Uint32Array(tensor.getShape()),
      (tensor.indices as WASMTensor<'uint32'>).wasmTensor,
      new Uint32Array(axes),
      keepDims
    ) as any,
    undefined,
    tensor.dtype
  );
}
