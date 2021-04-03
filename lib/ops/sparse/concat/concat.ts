import {CPUTensor} from '../../../tensor/cpu/tensor';
import {GPUTensor} from '../../../tensor/gpu/tensor';
import {SparseTensor} from '../../../tensor/sparse/tensor';
import {WASMTensor} from '../../../tensor/wasm/tensor';
import Tensor, {DType} from '../../../types';
import {compareShapes} from '../../../util/shape';
import {addIndexCPU} from './cpu';
import {addIndexGPU} from './gpu';
import {addIndexWASM} from './wasm';

export function concat<DTpe extends DType>(
  a: SparseTensor<DTpe>,
  b: SparseTensor<DTpe>,
  axis: number
) {
  if (!compareShapes(a.shape, b.shape) || a.sparseDims !== b.sparseDims) {
    throw new Error(
      'Sparse tensors can only be concatenated with the same shape and number of sparse dims'
    );
  }
  if (axis > a.sparseDims) {
    throw new Error(
      'Concatenation along dense axis of sparse tensor not supported yet'
    );
  } else {
    const values = a.values.concat(b.values, 0);
    const indexAdded = addIndex(b.indices, axis, a.shape[axis]);
    const indices = a.indices.concat(indexAdded, 0);
    indexAdded.delete();

    const resultShape = [...a.shape];
    resultShape[axis] += b.shape[axis];

    return new SparseTensor(values, indices, resultShape, a.denseDims);
  }
}

function addIndex(
  indices: Tensor<'uint32'>,
  axis: number,
  count: number
): Tensor<'uint32'> {
  if (indices instanceof CPUTensor) {
    return addIndexCPU(indices, axis, count);
  } else if (indices instanceof WASMTensor) {
    return addIndexWASM(indices, axis, count);
  } else {
    return addIndexGPU(indices as GPUTensor<'uint32'>, axis, count);
  }
}
