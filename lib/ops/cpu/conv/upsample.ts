import {CPUTensor} from '../../../tensor/cpu/tensor';
import {DType} from '../../../types';
import {incrementIndex} from '../../../util/shape';

export function upsample<DTpe extends DType>(
  x: CPUTensor<DTpe>,
  scales: number[]
) {
  const rank = x.shape.length;

  const resultShape = [...x.shape];
  for (let i = 0; i < rank; i++) {
    resultShape[i] = Math.floor(resultShape[i] * scales[i]);
  }

  const result = new CPUTensor(resultShape, undefined, x.dtype);

  const outIx = new Array(rank).fill(0);
  const inIx = new Array(rank);
  for (let i = 0; i < result.size; i++) {
    for (let j = 0; j < rank; j++) {
      inIx[j] = Math.floor(outIx[j] / scales[j]);
    }

    result.set(i, x.get(inIx));

    incrementIndex(outIx, resultShape);
  }

  return result;
}
