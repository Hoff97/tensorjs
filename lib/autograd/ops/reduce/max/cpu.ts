import {poolResultShape} from '../../../../ops/util/pool';
import {CPUTensor} from '../../../../tensor/cpu/tensor';
import {DType} from '../../../../types';

export function maxBackCPU<DTpe extends DType>(
  value: CPUTensor<DTpe>,
  gradient: CPUTensor<DTpe>,
  sumDims: number[]
) {
  const outShape = poolResultShape(value.shape, sumDims, false);
}
