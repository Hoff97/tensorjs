import {CPUTensor} from '../../../tensor/cpu/tensor';
import {DType} from '../../../types';
import {incrementIndex} from '../../../util/shape';

export function expand<DTpe extends DType>(
  x: CPUTensor<DTpe>,
  resultShape: readonly number[]
) {
  const rank = x.shape.length;

  const result = new CPUTensor(resultShape, undefined, x.dtype);

  const index = new Array(rank).fill(0);
  for (let i = 0; i < result.size; i++) {
    result.set(i, x.get(index));

    incrementIndex(index, result.shape);
  }

  return result;
}
