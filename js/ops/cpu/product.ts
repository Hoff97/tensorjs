import CPUTensor from '../../tensor/cpu/tensor';

import { pool } from './pool';

export function product(a: CPUTensor, axes: number[], keepDims: boolean): CPUTensor {
  return pool(a, axes, (a,b) => {
    return a * (b !== undefined ? b : 1);
  }, keepDims);
}