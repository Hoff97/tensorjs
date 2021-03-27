import {CPUTensor} from '../../../../tensor/cpu/tensor';
import {SparseTensor} from '../../../../tensor/sparse/tensor';
import {DType} from '../../../../types';
import {aggregateSparseCPU} from '../cpu';

export function reduceLogSumExpSparseCPU<DTpe extends DType>(
  tensor: SparseTensor<DTpe>,
  axes: number[],
  keepDims: boolean
): CPUTensor<DTpe> {
  return aggregateSparseCPU(
    tensor,
    axes,
    keepDims,
    (a, b) => a + Math.exp(b),
    e => Math.exp(e),
    (e, c) => Math.log(e)
  );
}