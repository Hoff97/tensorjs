import {positionWiseBinaryOp} from '../../../ops/cpu/basic';
import {CPUTensor} from '../../../tensor/cpu/tensor';

export function bce(x: CPUTensor, y: CPUTensor) {
  return positionWiseBinaryOp(
    x,
    y,
    (x: number, y: number) => {
      if (y === 1) {
        return -Math.log(x);
      } else {
        return -Math.log(1 - x);
      }
    },
    x.shape
  );
}
