import {Variable} from '../../../autograd';
import {Tensor} from '../../../library';
import {CPUTensor} from '../../../tensor/cpu/tensor';
import {GPUTensor} from '../../../tensor/gpu/tensor';
import {WASMTensor} from '../../../tensor/wasm/tensor';
import {sameType} from '../../../util/convert';
import {BCEBack} from './back/back';
import {bce as bceCPU} from './cpu';
import {defaultBCED} from './gpu';

export function bce(x: Tensor, y: Tensor): Tensor {
  if (!sameType(x, y)) {
    throw new Error('BCE can only be computed for tensors of the same type');
  }
  if (x instanceof CPUTensor && y instanceof CPUTensor) {
    return bceCPU(x, y);
  } else if (x instanceof WASMTensor && y instanceof WASMTensor) {
    return new WASMTensor(x.wasmTensor.bce(y.wasmTensor));
  } else if (x instanceof GPUTensor && y instanceof GPUTensor) {
    return defaultBCED.calc(
      {A: x as GPUTensor, B: y as GPUTensor, outputShape: x.getShape()},
      x.precision
    ) as GPUTensor;
  } else {
    return new Variable(bce((x as Variable).value, (y as Variable).value), {
      noGrad: (x as Variable).noGrad,
      backEdge: (x as Variable).noGrad
        ? undefined
        : new BCEBack(x as Variable, y as Variable),
    });
  }
}
