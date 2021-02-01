import {CPUTensor} from '../../tensor/cpu/tensor';

export function concat(x: CPUTensor, y: CPUTensor, axis: number) {
  const outputShape = [...x.shape];
  outputShape[axis] += y.shape[axis];

  const result = new CPUTensor(outputShape);

  let indexX = 0;
  let indexY = 0;

  let ix = 0;

  const iterXSize = result.strides[axis] * x.shape[axis];
  const iterYSize = result.strides[axis] * y.shape[axis];

  const outerIters =
    result.size / (axis > 0 ? result.strides[axis - 1] : result.size);
  for (let i = 0; i < outerIters; i++) {
    for (let j = 0; j < iterXSize; j++) {
      result.set(ix, x.get(indexX));
      ix++;
      indexX++;
    }

    for (let j = 0; j < iterYSize; j++) {
      result.set(ix, y.get(indexY));
      ix++;
      indexY++;
    }
  }

  return result;
}
