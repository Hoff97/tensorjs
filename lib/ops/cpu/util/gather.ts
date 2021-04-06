import {CPUTensor} from '../../../tensor/cpu/tensor';
import {DType} from '../../../types';
import {incrementIndex} from '../../../util/shape';

export function gather<DTpe extends DType>(
  x: CPUTensor<DTpe>,
  axis: number,
  indices: CPUTensor<'uint32'>
) {
  const r = x.shape.length;
  const q = indices.shape.length;

  const resultRank = r + q - 1;
  const resultShape = new Array(resultRank);
  for (let i = 0; i < axis; i++) {
    resultShape[i] = x.shape[i];
  }
  for (let i = 0; i < q; i++) {
    resultShape[i + axis] = indices.shape[i];
  }
  for (let i = axis + 1; i < r; i++) {
    resultShape[i + q - 1] = x.shape[i];
  }

  const result = new CPUTensor(resultShape, undefined, x.dtype);

  const outIx = new Array(resultRank).fill(0);
  let gatherIx: number[];
  let inputIx: number[];
  for (let i = 0; i < result.size; i++) {
    gatherIx = outIx.slice(axis, axis + q);
    const axIx = indices.get(gatherIx);
    inputIx = [...outIx.slice(0, axis), axIx, ...outIx.slice(axis + q)];

    result.set(i, x.get(inputIx));

    incrementIndex(outIx, resultShape);
  }

  return result;
}
