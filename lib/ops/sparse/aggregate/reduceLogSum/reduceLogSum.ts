import {Tensor} from '../../../../library';
import {CPUTensor} from '../../../../tensor/cpu/tensor';
import {SparseTensor} from '../../../../tensor/sparse/tensor';
import {WASMTensor} from '../../../../tensor/wasm/tensor';
import {DType} from '../../../../types';
import {poolResultShape} from '../../../util/pool';
import {reduceLogSumSparseCPU} from './cpu';
import {reduceLogSumSparseWASM} from './wasm';

export function reduceLogSum<DTpe extends DType>(
  tensor: SparseTensor<DTpe>,
  axes: number[],
  keepDims: boolean
) {
  if (axes.find(ax => ax < tensor.sparseDims) !== undefined) {
    return reduceLogSumSparse(tensor, axes, keepDims);
  } else {
    const [resultShape, _ixMap] = poolResultShape(tensor.shape, axes, keepDims);
    return new SparseTensor(
      tensor.values.reduceLogSum(
        axes.map(ax => ax - tensor.sparseDims + 1),
        keepDims
      ),
      tensor.indices.copy(),
      resultShape,
      keepDims ? tensor.denseDims : tensor.denseDims - axes.length
    );
  }
}

function reduceLogSumSparse<DTpe extends DType>(
  tensor: SparseTensor<DTpe>,
  axes: number[],
  keepDims: boolean
): Tensor<DTpe> {
  if (tensor.values instanceof CPUTensor) {
    return reduceLogSumSparseCPU(tensor, axes, keepDims);
  } else if (tensor.values instanceof WASMTensor) {
    return reduceLogSumSparseWASM(tensor as any, axes, keepDims) as any;
  }
  throw new Error(
    'Reduce log sum over sparse dimensions not implemented in WebGL yet'
  );
}
