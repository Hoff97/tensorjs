import {CPUTensor} from '../../../../tensor/cpu/tensor';
import {SparseTensor} from '../../../../tensor/sparse/tensor';
import {DType} from '../../../../types';
import {aggregateSparseCPU} from '../cpu';

export function sumSquareSparseCPU<DTpe extends DType>(
  tensor: SparseTensor<DTpe>,
  axes: number[],
  keepDims: boolean
): CPUTensor<DTpe> {
  return aggregateSparseCPU(
    tensor,
    axes,
    keepDims,
    (a, b) => a + b * b,
    e => e * e
  );
}
