import {CPUTensor} from '../../../../tensor/cpu/tensor';
import {SparseTensor} from '../../../../tensor/sparse/tensor';
import {DType} from '../../../../types';
import {aggregateSparseCPU} from '../cpu';

export function reduceMeanSparseCPU<DTpe extends DType>(
  tensor: SparseTensor<DTpe>,
  axes: number[],
  keepDims: boolean
): CPUTensor<DTpe> {
  return aggregateSparseCPU(
    tensor,
    axes,
    keepDims,
    (a, b) => a + b,
    undefined,
    (e, c) => e / c
  );
}
