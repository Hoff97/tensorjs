import {Tensor} from '../../../../library';
import {CPUTensor} from '../../../../tensor/cpu/tensor';
import {SparseTensor} from '../../../../tensor/sparse/tensor';
import {WASMTensor} from '../../../../tensor/wasm/tensor';
import {DType} from '../../../../types';
import {poolResultShape} from '../../../util/pool';
import {sumSparseCPU} from './cpu';
import {sumSparseWASM} from './wasm';

export function sum<DTpe extends DType>(
  tensor: SparseTensor<DTpe>,
  axes: number[],
  keepDims: boolean
) {
  if (axes.find(ax => ax < tensor.sparseDims) !== undefined) {
    return sumSparse(tensor, axes, keepDims);
  } else {
    const [resultShape, _ixMap] = poolResultShape(tensor.shape, axes, keepDims);
    return new SparseTensor(
      tensor.values.sum(
        axes.map(ax => ax - tensor.sparseDims + 1),
        keepDims
      ),
      tensor.indices.copy(),
      resultShape,
      keepDims ? tensor.denseDims : tensor.denseDims - axes.length
    );
  }
}

function sumSparse<DTpe extends DType>(
  tensor: SparseTensor<DTpe>,
  axes: number[],
  keepDims: boolean
): Tensor<DTpe> {
  if (tensor.values instanceof CPUTensor) {
    return sumSparseCPU(tensor, axes, keepDims);
  } else if (tensor.values instanceof WASMTensor) {
    return sumSparseWASM(tensor as any, axes, keepDims) as any;
  }
  throw new Error('Sum over sparse dimensions not implemented in WebGL yet');
}
