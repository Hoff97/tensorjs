import {positionWiseBinaryOp} from '../../../../ops/cpu/basic';
import {CPUTensor} from '../../../../tensor/cpu/tensor';

export function bceBack(x: CPUTensor, y: CPUTensor) {
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
