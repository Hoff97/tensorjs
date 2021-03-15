import {SparseTensor} from '../../../tensor/sparse/tensor';
import {DTypeWasm, WASMTensor} from '../../../tensor/wasm/tensor';
import {getSize} from '../../../util/shape';

export function reshapeWasm<DTpe extends DTypeWasm>(
  tensor: SparseTensor<DTpe>,
  values: WASMTensor<DTpe>,
  indices: WASMTensor<'uint32'>,
  shape: readonly number[],
  copy: boolean
): SparseTensor<DTpe> {
  const oldSparseSize = getSize(tensor.getSparseShape());

  const sparseShape = [];
  const denseShape = [];
  let sparseSize = 1;
  for (let i = 0; i < shape.length; i++) {
    if (sparseSize < oldSparseSize) {
      sparseSize *= shape[i];
      sparseShape.push(shape[i]);
    } else {
      denseShape.push(shape[i]);
    }
  }

  const nnzFraction = sparseSize / oldSparseSize;
  const nnz = tensor.nnz * nnzFraction;

  const newValues = values.reshape([nnz, ...denseShape], copy);
  const newIndices = new WASMTensor(
    indices.wasmTensor.reshape_sparse_indices(
      new Uint32Array(tensor.getSparseShape()),
      new Uint32Array(shape)
    ),
    undefined,
    'uint32'
  );

  return new SparseTensor(newValues, newIndices, shape, denseShape.length);
}
