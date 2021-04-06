import {CPUTensor} from '../../../tensor/cpu/tensor';
import {DType} from '../../../types';
import {incrementIndex} from '../../../util/shape';

export function transpose<DTpe extends DType>(
  x: CPUTensor<DTpe>,
  permutation: number[]
) {
  const rank = x.shape.length;

  const outputShape = new Array(rank);
  const reversePerm = new Array(rank);
  for (let i = 0; i < rank; i++) {
    outputShape[i] = x.shape[permutation[i]];
    reversePerm[permutation[i]] = i;
  }

  const result = new CPUTensor(outputShape, undefined, x.dtype);

  const resultStrides = result.strides;
  const mappedStrides = new Array(rank);
  for (let i = 0; i < rank; i++) {
    mappedStrides[i] = resultStrides[reversePerm[i]];
  }

  const index = new Array(rank).fill(0);
  for (let i = 0; i < x.size; i++) {
    let outIx = 0;
    for (let j = 0; j < rank; j++) {
      outIx += index[j] * mappedStrides[j];
    }

    result.set(outIx, x.get(i));

    incrementIndex(index, x.shape);
  }

  return result;
}
