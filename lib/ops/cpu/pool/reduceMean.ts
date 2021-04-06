import {CPUTensor} from '../../../tensor/cpu/tensor';
import {DType} from '../../../types';

import {pool} from './pool';

export function reduceMean<DTpe extends DType>(
  a: CPUTensor<DTpe>,
  axes: number[],
  keepDims: boolean
): CPUTensor<DTpe> {
  let poolSize = 1;
  for (let i = 0; i < axes.length; i++) {
    poolSize *= a.shape[axes[i]];
  }

  return pool(
    a,
    axes,
    (a, b) => {
      return a + (b !== undefined ? b : 0);
    },
    keepDims,
    (a: number) => a / poolSize
  );
}
