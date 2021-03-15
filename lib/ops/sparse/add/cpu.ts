import {CPUTensor} from '../../../tensor/cpu/tensor';
import {SparseTensor} from '../../../tensor/sparse/tensor';
import {DType} from '../../../types';
import {computeStrides, getSize, incrementIndex} from '../../../util/shape';

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

export function addSparseCPU<DTpe extends DType>(
  a: SparseTensor<DTpe>,
  b: SparseTensor<DTpe>,
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

  const valsB = b.values as CPUTensor<DTpe>;
  const indicesB = b.indices as CPUTensor<'uint32'>;

  const sparseStrides = computeStrides(sparseResultShape);

  const bIxPositionMap: {[position: number]: number} = {};
  for (let i = 0; i < b.nnz; i++) {
    let pos = 0;
    for (let j = 0; j < S; j++) {
      pos += indicesB.get(i * S + j) * sparseStrides[j];
    }
    bIxPositionMap[pos] = i;
  }

  const values = new CPUTensor(
    [a.nnz, ...denseResultShape],
    undefined,
    a.dtype
  );
  const indices = a.indices.copy();

  for (let i = 0; i < a.nnz; i++) {
    let pos = 0;
    for (let j = 0; j < S; j++) {
      pos += indicesA.get(i * S + j) * sparseStrides[j];
    }
    const iB = bIxPositionMap[pos];

    const denseIx: number[] = new Array(denseResultShape.length).fill(0);
    for (let j = 0; j < denseSize; j++) {
      const vA = valsA.get([i, ...denseIx]);
      const vB = valsB.get([iB, ...denseIx]);

      values.set(i * denseSize + j, alpha * vA + beta * vB);

      incrementIndex(denseIx, denseResultShape);
    }
  }

  return new SparseTensor(values, indices, resultShape, a.denseDims);
}
