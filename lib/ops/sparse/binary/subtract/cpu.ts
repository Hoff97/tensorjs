import {CPUTensor} from '../../../../tensor/cpu/tensor';
import {SparseTensor} from '../../../../tensor/sparse/tensor';
import {DType} from '../../../../types';
import {binaryDenseCPU, binarySparseCPU} from '../cpu';

export function subtractDenseCPU<DTpe extends DType>(
  a: SparseTensor<DTpe>,
  b: CPUTensor<DTpe>,
  resultShape: readonly number[],
  alpha: number,
  beta: number
): SparseTensor<DTpe> {
  return binaryDenseCPU(a, b, resultShape, (a, b) => alpha * a - beta * b);
}

export function subtractSparseCPU<DTpe extends DType>(
  a: SparseTensor<DTpe>,
  b: SparseTensor<DTpe>,
  resultShape: readonly number[],
  alpha: number,
  beta: number
): SparseTensor<DTpe> {
  return binarySparseCPU(a, b, resultShape, (a, b) => alpha * a - beta * b);
}
