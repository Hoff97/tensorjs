import {CPUTensor} from '../../tensor/cpu/tensor';
import {DType} from '../../types';

import {pool} from './pool';

export function reduceLogSum<DTpe extends DType>(
  a: CPUTensor<DTpe>,
  axes: number[],
  keepDims: boolean
): CPUTensor<DTpe> {
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
