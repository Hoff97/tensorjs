import {CPUTensor} from '../../tensor/cpu/tensor';
import {incrementIndex} from '../../util/shape';

export function normalize(
  x: CPUTensor,
  mean: CPUTensor,
  variance: CPUTensor,
  epsilon: number,
  scale: CPUTensor,
  bias: CPUTensor
) {
  const rank = x.shape.length;

  const resultShape = [...x.shape];

  const result = new CPUTensor(resultShape, undefined, x.type);

  const outIx = new Array(rank).fill(0);
  for (let i = 0; i < result.size; i++) {
    let res =
      (x.get(outIx) - mean.get(outIx)) /
      Math.sqrt(variance.get(outIx) + epsilon);

    res = res * scale.get(outIx) + bias.get(outIx);

    result.set(i, res);

    incrementIndex(outIx, resultShape);
  }

  return result;
}
