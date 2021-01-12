import { CPUTensor } from '../../tensor/cpu/tensor';
import { incrementIndex } from '../../util/shape';

export function slice(x: CPUTensor, starts: number[], ends: number[], axis: number[]) {
  const rank = x.shape.length;

  const resultShape = [...x.shape];
  let axIx = 0;
  for (let i = 0; i < rank && axIx < axis.length; i++) {
    if (i == axis[axIx]) {
      resultShape[i] = ends[axIx] - starts[axIx];
      axIx++;
    }
  }

  const result = new CPUTensor(resultShape, undefined, x.type);

  const outIx = new Array(rank).fill(0);
  let inIx: number[];

  for (let i = 0; i < result.size; i++) {
    inIx = [...outIx];
    for (let j = 0; j < axis.length; j++) {
      inIx[axis[j]] += starts[j];
    }

    result.set(i, x.get(inIx));

    incrementIndex(outIx, resultShape);
  }

  return result;
}
