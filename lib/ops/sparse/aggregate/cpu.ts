import {CPUTensor} from '../../../tensor/cpu/tensor';
import {SparseTensor} from '../../../tensor/sparse/tensor';
import {DType} from '../../../types';
import {getSize, incrementIndex, indexToPos} from '../../../util/shape';
import {poolResultShape} from '../../util/pool';

export function aggregateSparseCPU<DTpe extends DType>(
  tensor: SparseTensor<DTpe>,
  axes: number[],
  keepDims: boolean,
  op: (r: number, e: number) => number,
  init?: (e: number) => number
): CPUTensor<DTpe> {
  const [resultShape, ixMap] = poolResultShape(tensor.shape, axes, keepDims);
  const result = new CPUTensor(resultShape, undefined, tensor.values.dtype);

  const denseShape = tensor.getDenseShape();
  const denseSize = getSize(denseShape, 1);

  const resultSize = getSize(resultShape);
  let count: number[] | undefined;
  if (init !== undefined) {
    count = new Array(resultSize).fill(0);
  }

  const aIndices = tensor.indices as CPUTensor<'uint32'>;
  const aValues = tensor.values as CPUTensor<DTpe>;
  for (let i = 0; i < tensor.nnz; i++) {
    const sparseIx: number[] = new Array(tensor.sparseDims);
    for (let j = 0; j < tensor.sparseDims; j++) {
      sparseIx[j] = aIndices.get(i * tensor.sparseDims + j);
    }

    const denseIx: number[] = new Array(tensor.denseDims).fill(0);
    for (let j = 0; j < denseSize; j++) {
      const resultIx = [...sparseIx, ...denseIx];
      const mappedResultIx = new Array(ixMap.length);
      for (let k = 0; k < ixMap.length; k++) {
        mappedResultIx[k] = resultIx[ixMap[k]];
      }

      const pos = indexToPos(mappedResultIx, result.strides);

      if (init !== undefined && count !== undefined) {
        if (count[pos] === 0) {
          result.set(pos, init(aValues.get(i * denseSize + j)));
        } else {
          result.set(pos, op(result.get(pos), aValues.get(i * denseSize + j)));
        }
        count[pos]++;
      } else {
        result.set(pos, op(result.get(pos), aValues.get(i * denseSize + j)));
      }

      incrementIndex(denseIx, denseShape);
    }
  }

  return result;
}
