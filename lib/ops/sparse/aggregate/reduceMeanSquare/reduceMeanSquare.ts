import {Tensor} from '../../../../library';
import {CPUTensor} from '../../../../tensor/cpu/tensor';
import {SparseTensor} from '../../../../tensor/sparse/tensor';
import {DType} from '../../../../types';
import {poolResultShape} from '../../../util/pool';
import {reduceMeanSquareSparseCPU} from './cpu';

export function reduceMeanSquare<DTpe extends DType>(
  tensor: SparseTensor<DTpe>,
  axes: number[],
  keepDims: boolean
) {
  if (axes.find(ax => ax < tensor.sparseDims) !== undefined) {
    return reduceMeanSquareSparse(tensor, axes, keepDims);
  } else {
    const [resultShape, _ixMap] = poolResultShape(tensor.shape, axes, keepDims);
    return new SparseTensor(
      tensor.values.reduceMeanSquare(
        axes.map(ax => ax - tensor.sparseDims + 1),
        keepDims
      ),
      tensor.indices.copy(),
      resultShape,
      keepDims ? tensor.denseDims : tensor.denseDims - axes.length
    );
  }
}

function reduceMeanSquareSparse<DTpe extends DType>(
  tensor: SparseTensor<DTpe>,
  axes: number[],
  keepDims: boolean
): Tensor<DTpe> {
  if (tensor.values instanceof CPUTensor) {
    return reduceMeanSquareSparseCPU(tensor, axes, keepDims);
  }
  throw new Error(
    'Reduce mean squared over sparse dimensions not implemented in WASM/WebGL yet'
  );
}