import {poolResultShape} from '../../../../ops/util/pool';
import {CPUTensor} from '../../../../tensor/cpu/tensor';
import {DType} from '../../../../types';
import {incrementIndex} from '../../../../util/shape';

export function minBackCPU<DTpe extends DType>(
  value: CPUTensor<DTpe>,
  gradient: CPUTensor<DTpe>,
  axes: number[]
) {
  const [outShape, ixMap] = poolResultShape(value.shape, axes, false);

  const argMin = value.argMin(axes) as CPUTensor<'uint32'>;
  gradient = gradient.reshape(outShape, false) as CPUTensor<DTpe>;

  const result = new CPUTensor(value.shape, undefined, value.dtype);
  const resultIx = new Array(value.shape.length).fill(0);
  const gradIx = new Array(gradient.shape.length).fill(0);

  for (let i = 0; i < gradient.size; i++) {
    for (let j = 0; j < gradient.shape.length; j++) {
      resultIx[ixMap[j]] = gradIx[j];
    }
    for (let j = 0; j < axes.length; j++) {
      resultIx[axes[j]] = argMin.get(i * axes.length + j);
    }

    result.set(resultIx, gradient.get(i));

    incrementIndex(gradIx, gradient.shape);
  }

  return result;
}
