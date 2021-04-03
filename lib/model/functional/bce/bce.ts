import {Variable} from '../../../autograd';
import {Tensor} from '../../../library';
import {CPUTensor} from '../../../tensor/cpu/tensor';
import {GPUTensor} from '../../../tensor/gpu/tensor';
import {WASMTensor} from '../../../tensor/wasm/tensor';
import {DType} from '../../../types';
import {sameType} from '../../../util/convert';
import {BCEBack} from './back/back';
import {bce as bceCPU} from './cpu';
import {defaultBCED} from './gpu';

/**
 * Calculates the binary cross entropy loss, given probabilities x
 * and ground truth y. Returns a tensor of the same shape as
 * x. To use for a loss, you have to sum over the result:
 * ```typescript
 * const loss = bce(x,y).sum();
 * ```
 *
 * @param x Probabilities in [0,1]
 * @param y Ground truth labels of the same shape as x.
 */
export function bce<DTpe extends DType>(
  x: Tensor<DTpe>,
  y: Tensor<DTpe>
): Tensor<DTpe> {
  if (!sameType(x, y)) {
    throw new Error('BCE can only be computed for tensors of the same type');
  }
  if (x instanceof CPUTensor && y instanceof CPUTensor) {
    return bceCPU(x, y);
  } else if (x instanceof WASMTensor && y instanceof WASMTensor) {
    return new WASMTensor(x.wasmTensor.bce(y.wasmTensor)) as any;
  } else if (x instanceof GPUTensor && y instanceof GPUTensor) {
    return defaultBCED.calc(
      {
        A: x as any,
        B: y as any,
        outputShape: x.getShape(),
      },
      x.dtype
    ) as any;
  } else {
    return new Variable(
      bce((x as Variable<DTpe>).value, (y as Variable<DTpe>).value),
      {
        noGrad: (x as Variable<DTpe>).noGrad,
        backEdge: (x as Variable<DTpe>).noGrad
          ? undefined
          : new BCEBack(x as Variable<DTpe>, y as Variable<DTpe>),
      }
    );
  }
}
