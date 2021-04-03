import {CPUTensor} from '../../../../tensor/cpu/tensor';
import {SparseTensor} from '../../../../tensor/sparse/tensor';
import {DType} from '../../../../types';
import {binaryDenseCPU, binarySparseCPU} from '../cpu';

export function multiplyDenseCPU<DTpe extends DType>(
  a: SparseTensor<DTpe>,
  b: CPUTensor<DTpe>,
  resultShape: readonly number[],
  alpha: number
): SparseTensor<DTpe> {
  return binaryDenseCPU(a, b, resultShape, (a, b) => alpha * a * b);
}

export function multiplySparseCPU<DTpe extends DType>(
  a: SparseTensor<DTpe>,
  b: SparseTensor<DTpe>,
  resultShape: readonly number[],
  alpha: number
): SparseTensor<DTpe> {
  return binarySparseCPU(a, b, resultShape, (a, b) => alpha * a * b);
}
