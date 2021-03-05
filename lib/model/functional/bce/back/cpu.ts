import {positionWiseBinaryOp} from '../../../../ops/cpu/basic';
import {CPUTensor} from '../../../../tensor/cpu/tensor';
import {DType} from '../../../../types';

export function bceBack<DTpe extends DType>(
  x: CPUTensor<DTpe>,
  y: CPUTensor<DTpe>
) {
  return positionWiseBinaryOp(
    x,
    y,
    (x: number, y: number) => {
      if (y === 1) {
        return -1 / x;
      } else {
        return 1 / (1 - x);
      }
    },
    x.shape
  );
}
