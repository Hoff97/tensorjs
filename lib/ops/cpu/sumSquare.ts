import {CPUTensor} from '../../tensor/cpu/tensor';

import {pool} from './pool';

export function sumSquare(
  a: CPUTensor,
  axes: number[],
  keepDims: boolean
): CPUTensor {
  return pool(
    a,
    axes,
    (a, b) => {
      return a * a + (b !== undefined ? b : 0);
    },
    keepDims
  );
}
