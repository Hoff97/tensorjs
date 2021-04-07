import {poolResultShape} from '../../../../ops/util/pool';
import {DTypeGpu} from '../../../../tensor/gpu/interface';
import {GPUTensor} from '../../../../tensor/gpu/tensor';
import {defaultMaxBackD} from '../max/gpu';

export function minBackGPU<DTpe extends DTypeGpu>(
  value: GPUTensor<DTpe>,
  gradient: GPUTensor<DTpe>,
  axes: number[]
) {
  let argMin = value.argMin(axes, false);

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [resultShape, _] = poolResultShape(value.shape, axes, true);
  gradient = gradient.reshape(resultShape, false) as GPUTensor<DTpe>;
  argMin = argMin.reshape([...resultShape, axes.length], false);

  return defaultMaxBackD.calc(
    {
      Grad: gradient,
      ArgMax: argMin,
      axes,
      valueShape: value.shape,
    },
    value.dtype
  ) as GPUTensor<DTpe>;
}
