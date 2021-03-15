import {CPUTensor} from '../../../tensor/cpu/tensor';
import {SparseTensor} from '../../../tensor/sparse/tensor';
import {DType} from '../../../types';
import {getSize, incrementIndex} from '../../../util/shape';

export function addDenseCPU<DTpe extends DType>(
  a: SparseTensor<DTpe>,
  b: CPUTensor<DTpe>,
  resultShape: readonly number[],
  alpha: number,
  beta: number
): SparseTensor<DTpe> {
  const S = a.sparseDims;

  const sparseResultShape = resultShape.slice(0, S);
  const denseResultShape = resultShape.slice(S);
  const denseSize = getSize(denseResultShape, 1);

  const valsA = a.values as CPUTensor<DTpe>;
  const indicesA = a.indices as CPUTensor<'uint32'>;

  const values = new CPUTensor(
    [a.nnz, ...denseResultShape],
    undefined,
    a.dtype
  );
  const indices = a.indices.copy();

  for (let i = 0; i < a.nnz; i++) {
    const sparseIx: number[] = new Array(sparseResultShape.length);
    for (let j = 0; j < sparseResultShape.length; j++) {
      sparseIx[j] = indicesA.get(i * S + j);
    }

    const denseIx: number[] = new Array(denseResultShape.length).fill(0);
    for (let j = 0; j < denseSize; j++) {
      const vA = valsA.get([i, ...denseIx]);
      const vB = b.get([...sparseIx, ...denseIx]);

      values.set(i * denseSize + j, alpha * vA + beta * vB);

      incrementIndex(denseIx, denseResultShape);
    }
  }

  return new SparseTensor(values, indices, resultShape, a.denseDims);
}
