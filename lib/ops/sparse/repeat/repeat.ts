import {CPUTensor} from '../../../tensor/cpu/tensor';
import {GPUTensor} from '../../../tensor/gpu/tensor';
import {SparseTensor} from '../../../tensor/sparse/tensor';
import {WASMTensor} from '../../../tensor/wasm/tensor';
import Tensor, {DType} from '../../../types';
import {repeatIndicesCPU} from './cpu';

export function repeat<DTpe extends DType>(
  tensor: SparseTensor<DTpe>,
  repeats: readonly number[]
): SparseTensor<DTpe> {
  const sparseRepeats = repeats.slice(0, tensor.sparseDims);
  const denseRepeats = repeats.slice(tensor.sparseDims);

  const sparseRepeatsProd = sparseRepeats.reduce((a, b) => a * b, 1);

  const values = tensor.values.repeat([sparseRepeatsProd, ...denseRepeats]);
  let indices: Tensor<'uint32'>;
  if (sparseRepeatsProd > 1) {
    indices = repeatIndices(
      tensor.indices,
      sparseRepeats,
      tensor.getSparseShape(),
      sparseRepeatsProd
    );
  } else {
    indices = tensor.indices.copy();
  }

  const newShape = tensor.shape.map((v, i) => v * repeats[i]);

  return new SparseTensor(values, indices, newShape, tensor.denseDims);
}

function repeatIndices(
  indices: Tensor<'uint32'>,
  repeats: readonly number[],
  shape: readonly number[],
  repeatsProd: number
): Tensor<'uint32'> {
  if (indices instanceof CPUTensor) {
    return repeatIndicesCPU(indices, repeats, shape, repeatsProd);
  }
  throw new Error('Repeat not implemented on backend WASM/GPU');
}
