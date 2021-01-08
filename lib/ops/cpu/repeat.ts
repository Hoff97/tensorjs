import { CPUTensor } from '../../tensor/cpu/tensor';
import { incrementIndex } from '../../util/shape';

export function repeat(x: CPUTensor, repeats: number[]) {
  const rank = x.shape.length;

  const outputShape = new Array(rank);
  for (let i = 0; i < rank; i++) {
    outputShape[i] = x.shape[i]*repeats[i];
  }

  const result = new CPUTensor(outputShape);

  const index = new Array(rank).fill(0);
  for (let i = 0; i < result.size; i++) {
    let inIndex = new Array(rank);
    for (let j = 0; j < rank; j++) {
      inIndex[j] = index[j] % x.shape[j];
    }

    result.set(i, x.get(inIndex));

    incrementIndex(index, result.shape);
  }

  return result;
}
