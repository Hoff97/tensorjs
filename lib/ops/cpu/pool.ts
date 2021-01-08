import CPUTensor from '../../tensor/cpu/tensor';
import { getSize, incrementIndex, indexToPos, computeStrides } from '../../util/shape';

import { poolResultShape } from '../util/pool';

export function pool(a: CPUTensor,
                     axes: number[],
                     operation: (a: number, b?: number) => number,
                     keepDims: boolean,
                     postProcess?: (a: number) => number): CPUTensor {
  const inputShape = a.getShape();
  const inputSize = getSize(inputShape);
  const [resultShape, ixMap] = poolResultShape(inputShape, axes, keepDims);
  const resultSize = getSize(resultShape);
  const resultStrides = computeStrides(resultShape);

  const result = new CPUTensor(resultShape);
  const initialized = new Array(resultSize).fill(false);

  const index: number[] = new Array(inputShape.length).fill(0);
  const outIndex: number[] = new Array(resultShape.length).fill(0);
  for (let i = 0; i < inputSize; i++) {
    for (let j = 0; j < ixMap.length; j++) {
      outIndex[j] = index[ixMap[j]];
    }
    const outOffset = indexToPos(outIndex, resultStrides);

    if (initialized[outOffset]) {
      result.set(outIndex, operation(a.get(i), result.get(outIndex)));
    } else {
      initialized[outOffset] = true;
      result.set(outIndex, operation(a.get(i)));
    }

    incrementIndex(index, inputShape);
  }

  if (postProcess) {
    for (let i = 0; i < result.size; i++) {
      result.set(i, postProcess(result.get(i)));
    }
  }

  return result;
}
