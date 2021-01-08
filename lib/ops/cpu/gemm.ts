import CPUTensor from '../../tensor/cpu/tensor';
import { getSize } from '../../util/shape';

export function gemm(a: CPUTensor, b: CPUTensor, aTranspose: boolean,
                     bTranspose: boolean, alpha: number, beta: number,
                     c?: CPUTensor) {
  const rank = a.shape.length;

  const M = aTranspose ? a.shape[rank - 1] : a.shape[rank - 2];
  const N = aTranspose ? a.shape[rank - 2] : a.shape[rank - 1];
  const O = bTranspose ? b.shape[rank - 2] : b.shape[rank - 1];

  const aBatchMult = M*N;
  const bBatchMult = N*O;
  const yBatchMult = M*O;

  const aNMult = aTranspose ? M : 1;
  const aMMult = aTranspose ? 1 : N;
  const bNMult = bTranspose ? 1 : O;
  const bOMult = bTranspose ? N : 1;

  let cMMult = 0;
  let cOMult = 0;
  let cBatchMult = 0;
  if (c !== undefined) {
    cMMult = c.strides[rank - 2];
    cOMult = c.strides[rank - 1];

    let cBatchSize = getSize(c.shape.slice(0, rank-2));
    if (cBatchSize > 1) {
      cBatchMult = c.shape[rank-2]*c.shape[rank-1];

      if (cBatchMult === 1) {
        cBatchMult = 0;
      }
    } else {
      cBatchMult = 0;
    }
  }

  const batchShape = a.shape.slice(0, rank-2);
  let batchSize = getSize(batchShape);
  if (batchSize === 0) {
    batchSize = 1;
  }
  const resultShape = [...batchShape, M, O];

  const Y = new CPUTensor(resultShape);

  for (let i = 0; i < batchSize; i++) {
    const aBase = i*aBatchMult;
    const bBase = i*bBatchMult;
    const yBase = i*yBatchMult;
    const cBase = i*cBatchMult;

    for (let m = 0; m < M; m++) {
      for (let o = 0; o < O; o++) {
        let result = 0;

        for (let n = 0; n < N; n++) {
          result += a.get(aBase + m*aMMult + n*aNMult) * b.get(bBase + n*bNMult + o*bOMult);
        }

        result = alpha*result;
        if (c !== undefined) {
          result += beta*c.get(cBase + m*cMMult + o*cOMult);
        }

        Y.set(yBase + m*O + o, result);
      }
    }
  }

  return Y;
}
