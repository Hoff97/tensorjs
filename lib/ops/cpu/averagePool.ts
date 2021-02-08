import {CPUTensor} from '../../tensor/cpu/tensor';
import {getSize, incrementIndex} from '../../util/shape';
import {outputDimsSize} from '../util/conv';

export function averagePool(
  x: CPUTensor,
  kernelShape: number[],
  pads: number[],
  strides: number[],
  includePad: boolean
) {
  const N = x.shape[0];
  const C = x.shape[1];
  const D = x.shape.slice(2);

  const dataRank = D.length;

  const kernelSize = getSize(kernelShape);

  const R = outputDimsSize(
    D,
    kernelShape,
    pads.slice(0, pads.length / 2),
    pads.slice(pads.length / 2),
    new Array(dataRank).fill(1),
    strides
  );
  const outputSize = getSize(R);
  let outputShape = [N, C];
  outputShape = outputShape.concat(R);

  const Y = new CPUTensor(outputShape);

  // Iterate over all batches
  for (let n = 0; n < N; n++) {
    // Iterate over all output channels
    for (let c = 0; c < C; c++) {
      const outputIndices = new Array(R.length).fill(0);
      outputIndices.unshift(n, c);
      for (let oIx = 0; oIx < outputSize; oIx++) {
        let result = 0;

        const kernelIndices: number[] = new Array(R.length).fill(0);

        let count = 0;

        for (let kIx = 0; kIx < kernelSize; kIx++) {
          const inputIx = [n, c];

          let skip = false;
          for (let axis = 0; axis < dataRank; axis++) {
            const stride = strides.length === 0 ? 1 : strides[axis];
            const pad = pads.length === 0 ? 0 : pads[axis];

            const ix =
              outputIndices[axis + 2] * stride - pad + kernelIndices[axis];

            if (ix < 0 || ix >= D[axis]) {
              skip = true;
              break;
            }

            inputIx.push(ix);
          }

          if (!skip) {
            const Xi = x.get(inputIx) as number;
            result += Xi;
          }

          if (!skip || includePad) {
            count += 1;
          }

          incrementIndex(kernelIndices, kernelShape);
        }

        result = result / count;

        Y.set(outputIndices, result);

        incrementIndex(outputIndices, Y.shape);
      }
    }
  }

  return Y;
}
