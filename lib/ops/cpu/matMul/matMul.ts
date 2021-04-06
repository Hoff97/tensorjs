import {CPUTensor} from '../../../tensor/cpu/tensor';
import {DType} from '../../../types';

export function matMul<DTpe extends DType>(
  a: CPUTensor<DTpe>,
  b: CPUTensor<DTpe>
) {
  if (a.shape.length !== 2 || b.shape.length !== 2) {
    throw new Error('Matmul expects both operands to have rank 2');
  }

  if (a.shape[1] !== b.shape[0]) {
    throw new Error(
      'Matmul expects dimension 1 of operand 1 to equal dimension 0 of operand 2'
    );
  }

  const m = a.shape[0];
  const n = a.shape[1];
  const o = b.shape[1];

  const result = new CPUTensor([m, o], undefined, a.dtype);

  for (let i = 0; i < m; i += 1) {
    for (let k = 0; k < o; k += 1) {
      let res = 0;
      for (let j = 0; j < n; j += 1) {
        res += a.get([i, j]) * b.get([j, k]);
      }
      result.set([i, k], res);
    }
  }

  return result;
}
