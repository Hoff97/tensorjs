import {CPUTensor} from '../../../tensor/cpu/tensor';
import {DType} from '../../../types';
import {incrementIndex} from '../../../util/shape';

export function slice<DTpe extends DType>(
  x: CPUTensor<DTpe>,
  starts: number[],
  ends: number[],
  axis: number[],
  steps: number[]
) {
  const rank = x.shape.length;

  const resultShape = [...x.shape];
  let axIx = 0;
  for (let i = 0; i < rank && axIx < axis.length; i++) {
    if (i === axis[axIx]) {
      resultShape[i] = Math.ceil((ends[axIx] - starts[axIx]) / steps[axIx]);
      axIx++;
    }
  }

  const result = new CPUTensor(resultShape, undefined, x.dtype);

  const outIx = new Array(rank).fill(0);
  let inIx: number[];

  for (let i = 0; i < result.size; i++) {
    inIx = new Array(rank);
    for (let j = 0; j < axis.length; j++) {
      inIx[axis[j]] = outIx[axis[j]] * steps[j] + starts[j];
    }

    result.set(i, x.get(inIx));

    incrementIndex(outIx, resultShape);
  }

  return result;
}
