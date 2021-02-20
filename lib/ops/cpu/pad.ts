import {CPUTensor} from '../../tensor/cpu/tensor';
import {DType, PadMode} from '../../types';
import {incrementIndex} from '../../util/shape';

export function pad<DTpe extends DType>(
  x: CPUTensor<DTpe>,
  pads: number[],
  mode: PadMode,
  value: number
) {
  const rank = x.shape.length;

  const resultShape = [...x.shape];
  for (let i = 0; i < rank; i++) {
    resultShape[i] += pads[i] + pads[i + rank];
  }

  const Y = new CPUTensor(resultShape, undefined, x.dtype);

  const ix = new Array(rank).fill(0);
  const inputIx = new Array(rank).fill(0);
  for (let i = 0; i < Y.size; i++) {
    let allInRange = true;
    for (let j = 0; j < rank; j++) {
      inputIx[j] = ix[j] - pads[j];
      if (inputIx[j] < 0 || inputIx[j] >= x.shape[j]) {
        allInRange = false;
      }
    }

    Y.set(i, getPadValue(x, inputIx, mode, value, allInRange));

    incrementIndex(ix, resultShape);
  }

  return Y;
}

function getPadValue(
  x: CPUTensor,
  index: number[],
  mode: PadMode,
  value: number,
  allInRange: boolean
): number {
  if (allInRange) {
    return x.get(index);
  }

  const rank = x.shape.length;

  if (mode === 'constant') {
    return value;
  } else if (mode === 'edge') {
    for (let j = 0; j < rank; j++) {
      if (index[j] < 0) {
        index[j] = 0;
      } else if (index[j] >= x.shape[j]) {
        index[j] = x.shape[j] - 1;
      }
    }
  } else {
    for (let j = 0; j < rank; j++) {
      if (index[j] < 0) {
        index[j] = -index[j];
      } else if (index[j] >= x.shape[j]) {
        index[j] = 2 * x.shape[j] - index[j] - 2;
      }
    }
  }

  return x.get(index);
}
