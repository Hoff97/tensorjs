import {CPUTensor} from '../../../tensor/cpu/tensor';
import {DType} from '../../../types';
import {
  getSize,
  incrementIndex,
  indexToPos,
  computeStrides,
} from '../../../util/shape';

import {poolResultShape} from '../../util/pool';

export function argMin<DTpe extends DType>(
  a: CPUTensor<DTpe>,
  axes: number[],
  selectLast: boolean
): CPUTensor<'uint32'> {
  const inputShape = a.getShape();
  const inputSize = getSize(inputShape);
  const [resultShape, ixMap] = poolResultShape(inputShape, axes, false);
  const resultSize = getSize(resultShape);
  const resultStrides = computeStrides(resultShape);

  const result = new CPUTensor(
    [...resultShape, axes.length],
    undefined,
    'uint32'
  );
  const values = new CPUTensor(resultShape, undefined, a.dtype);
  const initialized = new Array(resultSize).fill(false);

  const index: number[] = new Array(inputShape.length).fill(0);
  const outIndex: number[] = new Array(resultShape.length).fill(0);
  for (let i = 0; i < inputSize; i++) {
    for (let j = 0; j < ixMap.length; j++) {
      outIndex[j] = index[ixMap[j]];
    }
    const outOffset = indexToPos(outIndex, resultStrides);

    if (initialized[outOffset]) {
      const val = a.get(i);
      const existing = values.get(outOffset);

      if (val < existing || (selectLast && val === existing)) {
        values.set(outOffset, a.get(i));
        for (let j = 0; j < axes.length; j++) {
          result.set(outOffset * axes.length + j, index[axes[j]]);
        }
      }
    } else {
      initialized[outOffset] = true;
      values.set(outOffset, a.get(i));
      for (let j = 0; j < axes.length; j++) {
        result.set(outOffset * axes.length + j, index[axes[j]]);
      }
    }

    incrementIndex(index, inputShape);
  }

  return result;
}
