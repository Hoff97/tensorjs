import {CPUTensor} from '../../../tensor/cpu/tensor';
import {computeStrides, posToIndex} from '../../../util/shape';

export function repeatIndicesCPU(
  indices: CPUTensor<'uint32'>,
  repeats: readonly number[],
  shape: readonly number[],
  repeatsProd: number
): CPUTensor<'uint32'> {
  const nnz = indices.shape[0];
  const nnzNew = nnz * repeatsProd;
  const S = indices.shape[1];
  const result = new CPUTensor([nnz * repeatsProd, S], undefined, 'uint32');

  const repeatStrides = computeStrides(repeats);

  for (let i = 0; i < nnzNew; i++) {
    const oldI = i % nnz;
    const repeatPos = Math.floor(i / nnz);
    const repeatIx = posToIndex(repeatPos, repeatStrides);

    const ix = repeatIx.map((v, i) => v * shape[i]);

    for (let j = 0; j < S; j++) {
      result.set(i * S + j, ix[j] + indices.get(oldI * S + j));
    }
  }

  return result;
}
