import {CPUTensor} from '../../tensor/cpu/tensor';
import {incrementIndex} from '../../util/shape';

export function setValues(
  a: CPUTensor,
  b: CPUTensor,
  starts: readonly number[]
) {
  const result = new CPUTensor(a.shape);

  const index = new Array(a.shape.length).fill(0);

  for (let i = 0; i < result.size; i += 1) {
    let inB = true;
    const bIx = new Array(starts.length).fill(0);
    for (let j = 0; j < starts.length; j++) {
      if (index[j] < starts[j] || index[j] >= starts[j] + b.shape[j]) {
        inB = false;
        break;
      } else {
        bIx[j] = index[j] - starts[j];
      }
    }

    if (inB) {
      result.set(i, b.get(bIx));
    } else {
      result.set(i, a.get(i));
    }

    incrementIndex(index, a.shape);
  }

  return result;
}
