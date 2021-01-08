import { CPUTensor } from '../../tensor/cpu/tensor';

import { pool } from './pool';

export function sum(a: CPUTensor, axes: number[], keepDims: boolean): CPUTensor {
  return pool(a, axes, (a,b) => {
    return a + (b !== undefined ? b : 0);
  }, keepDims);
}