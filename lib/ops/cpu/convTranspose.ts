import {CPUTensor} from '../../tensor/cpu/tensor';
import {DType} from '../../types';
import {getSize, incrementIndex} from '../../util/shape';
import {outputDimsSize} from '../util/convTranspose';

export function convTranspose<DTpe extends DType>(
  x: CPUTensor<DTpe>,
  w: CPUTensor<DTpe>,
  dilations: number[],
  group: number,
  pads: number[],
  strides: number[]
) {
  const N = x.shape[0];
  const C = x.shape[1];
  const D = x.shape.slice(2);
  const W = w.shape.slice(2);
  const M = w.shape[0];
  const CG = C / group;

  const kernelSize = getSize(W);

  const R = outputDimsSize(
    D,
    W,
    pads.slice(0, pads.length / 2),
    pads.slice(pads.length / 2),
    dilations,
    strides
  );
  const outputSize = getSize(R);
  let outputShape = [N, M];
  outputShape = outputShape.concat(R);

  const Y = new CPUTensor(outputShape, undefined, x.dtype);

  const dataRank = R.length;

  // Iterate over all batches
  for (let n = 0; n < N; n++) {
    // Iterate over all output channels
    for (let m = 0; m < M; m++) {
      for (let cg = 0; cg < CG; cg++) {
        const c = (m * CG + cg) % C;

        const outputIndices = new Array(R.length).fill(0);
        outputIndices.unshift(n, m);
        for (let oIx = 0; oIx < outputSize; oIx++) {
          let result = Y.get(outputIndices) as number;

          const kernelIndices: number[] = new Array(R.length).fill(0);
          kernelIndices.unshift(m, cg);
          for (let kIx = 0; kIx < kernelSize; kIx++) {
            const inputIx = [n, c];

            let skip = false;
            for (let axis = 0; axis < dataRank; axis++) {
              const stride = strides.length === 0 ? 1 : strides[axis];
              const pad = pads.length === 0 ? 0 : pads[axis];
              const dilation = dilations.length === 0 ? 1 : dilations[axis];

              let ix =
                outputIndices[axis + 2] -
                pad +
                kernelIndices[axis + 2] * dilation;

              const res = ix % stride;

              if (res !== 0) {
                skip = true;
                break;
              }
              ix = ix / stride;

              if (ix < 0 || ix >= D[axis]) {
                skip = true;
                break;
              }

              inputIx.push(ix);
            }

            if (!skip) {
              const transposedKernelIx = [m, cg];
              for (let i = 0; i < dataRank; i++) {
                transposedKernelIx.push(W[i] - kernelIndices[i + 2] - 1);
              }
              const Wi = w.get(transposedKernelIx) as number;
              const Xi = x.get(inputIx) as number;
              result += Wi * Xi;
            }

            incrementIndex(kernelIndices, w.shape);
          }

          Y.set(outputIndices, result);

          incrementIndex(outputIndices, Y.shape);
        }
      }
    }
  }

  return Y;
}
