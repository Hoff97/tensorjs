import CPUTensor from '../../tensor/cpu/tensor';
import { getSize, incrementIndex } from '../../util/shape';

import { sumResultShape } from '../util/sum';

export function sum(a: CPUTensor, axes: number[]): CPUTensor {
  const inputShape = a.getShape();
  const inputSize = getSize(inputShape);
  const [resultShape, ixMap] = sumResultShape(inputShape, axes);

  const result = new CPUTensor(resultShape);

  const index = new Array(inputShape.length);
  for (let i = 0; i < inputShape.length; i++) {
    index[i] = 0;
  }
  const outIndex = new Array(resultShape.length);
  for (let i = 0; i < resultShape.length; i++) {
    outIndex[i] = 0;
  }
  for (let i = 0; i < inputSize; i++) {
    for (let j = 0; j < ixMap.length; j++) {
      outIndex[j] = index[ixMap[j]];
    }

    result.set(outIndex, result.get(outIndex) + a.get(i));

    incrementIndex(index, inputShape);
  }

  return result;
}
