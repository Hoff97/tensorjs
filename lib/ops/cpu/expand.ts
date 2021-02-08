import {CPUTensor} from '../../tensor/cpu/tensor';
import {incrementIndex} from '../../util/shape';

export function expand(x: CPUTensor, resultShape: readonly number[]) {
  const rank = x.shape.length;

  const result = new CPUTensor(resultShape);

  const index = new Array(rank).fill(0);
  for (let i = 0; i < result.size; i++) {
    result.set(i, x.get(index));

    incrementIndex(index, result.shape);
  }

  return result;
}
