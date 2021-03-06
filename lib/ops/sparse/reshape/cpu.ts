import {CPUTensor} from '../../../tensor/cpu/tensor';
import {SparseTensor} from '../../../tensor/sparse/tensor';
import {DType} from '../../../types';
import {
  compareShapes,
  computeStrides,
  getSize,
  indexToPos,
  posToIndex,
} from '../../../util/shape';

export function reshapeCPU<DTpe extends DType>(
  tensor: SparseTensor<DTpe>,
  values: CPUTensor<DTpe>,
  indices: CPUTensor<'uint32'>,
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

  const oldSparseStrides = computeStrides(tensor.getSparseShape());
  const newSparseStrides = computeStrides(sparseShape);

  const nnzFraction = sparseSize / oldSparseSize;
  const nnz = tensor.nnz * nnzFraction;

  const newValues = values.reshape([nnz, ...denseShape], copy);

  let newIndices: CPUTensor<'uint32'>;
  if (!copy && compareShapes(sparseShape, tensor.getSparseShape())) {
    newIndices = indices;
  } else {
    newIndices = new CPUTensor([nnz, sparseShape.length], undefined, 'uint32');
    for (let i = 0; i < nnz; i++) {
      const oldNnzIx = Math.floor(i / nnzFraction);

      const oldSparseIx = [];
      for (let j = 0; j < tensor.sparseDims; j++) {
        oldSparseIx.push(indices.get(oldNnzIx * tensor.sparseDims + j));
      }
      const oldSparsePos = indexToPos(oldSparseIx, oldSparseStrides);
      const newSparsePos = oldSparsePos * nnzFraction + (i % nnzFraction);
      const newSparseIx = posToIndex(newSparsePos, newSparseStrides);
      for (let j = 0; j < sparseShape.length; j++) {
        newIndices.set(i * sparseShape.length + j, newSparseIx[j]);
      }
    }
  }

  return new SparseTensor(newValues, newIndices, shape, denseShape.length);
}
