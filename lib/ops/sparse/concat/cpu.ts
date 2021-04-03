import {CPUTensor} from '../../../tensor/cpu/tensor';

export function addIndexCPU(
  indices: CPUTensor<'uint32'>,
  axis: number,
  count: number
): CPUTensor<'uint32'> {
  const result = indices.copy() as CPUTensor<'uint32'>;
  for (let i = axis; i < result.size; i += indices.shape[1]) {
    result.set(i, result.get(i) + count);
  }
  return result;
}
