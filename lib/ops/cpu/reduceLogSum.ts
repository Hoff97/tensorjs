import {CPUTensor} from '../../tensor/cpu/tensor';

import {pool} from './pool';

export function reduceLogSum(
  a: CPUTensor,
  axes: number[],
  keepDims: boolean
): CPUTensor {
  return pool(
    a,
    axes,
    (a, b) => {
      return a + (b !== undefined ? b : 0);
    },
    keepDims,
    (a: number) => Math.log(a)
  );
}
