import CPUTensor from '../../tensor/cpu/tensor';

import { pool } from './pool';

export function max(a: CPUTensor, axes: number[], keepDims: boolean): CPUTensor {
  return pool(a, axes, (a,b) => {
    return Math.max(a, b !== undefined ? b : a);
  }, keepDims);
}