import {CPUTensor} from '../../tensor/cpu/tensor';
import {checkEquivShapes, incrementIndex} from '../../util/shape';

// eslint-disable-next-line no-unused-vars
type UnaryOperator = (o: number) => number;
// eslint-disable-next-line no-unused-vars
type BinaryOperator = (o1: number, o2: number) => number;

export function positionWiseUnaryOp(a: CPUTensor, op: UnaryOperator) {
  const result = new CPUTensor(a.shape);

  for (let i = 0; i < result.size; i += 1) {
    result.set(i, op(a.get(i)));
  }

  return result;
}

export function positionWiseBinaryOp(
  a: CPUTensor,
  b: CPUTensor,
  op: BinaryOperator,
  resultShape: readonly number[]
) {
  if (!checkEquivShapes(a.shape, b.shape)) {
    throw new Error(
      'The shapes of the two tensors should be the same for a binary operation'
    );
  }

  const result = new CPUTensor(resultShape);

  const index = new Array(resultShape.length).fill(0);

  for (let i = 0; i < result.size; i += 1) {
    result.set(index, op(a.get(index), b.get(index)));

    incrementIndex(index, resultShape);
  }

  return result;
}

export function exp(a: CPUTensor) {
  return positionWiseUnaryOp(a, o1 => Math.exp(o1));
}

export function log(a: CPUTensor) {
  return positionWiseUnaryOp(a, o1 => Math.log(o1));
}

export function sqrt(a: CPUTensor) {
  return positionWiseUnaryOp(a, o1 => Math.sqrt(o1));
}

export function abs(a: CPUTensor) {
  return positionWiseUnaryOp(a, o1 => Math.abs(o1));
}

export function floor(a: CPUTensor) {
  return positionWiseUnaryOp(a, o1 => Math.floor(o1));
}

export function ceil(a: CPUTensor) {
  return positionWiseUnaryOp(a, o1 => Math.ceil(o1));
}

export function sign(a: CPUTensor) {
  return positionWiseUnaryOp(a, o1 => (o1 < 0 ? -1 : 1));
}

export function negate(a: CPUTensor) {
  return positionWiseUnaryOp(a, o1 => -o1);
}

export function clip(a: CPUTensor, min?: number, max?: number) {
  let f = (o1: number) => o1;
  if (min !== undefined && max !== undefined) {
    f = (o1: number) => Math.min(max, Math.max(min, o1));
  } else if (max !== undefined) {
    f = (o1: number) => Math.min(max, o1);
  } else if (min !== undefined) {
    f = (o1: number) => Math.max(min, o1);
  }

  return positionWiseUnaryOp(a, f);
}

export function add(
  a: CPUTensor,
  b: CPUTensor,
  resultShape: readonly number[]
) {
  return positionWiseBinaryOp(a, b, (o1, o2) => o1 + o2, resultShape);
}

export function subtract(
  a: CPUTensor,
  b: CPUTensor,
  resultShape: readonly number[]
) {
  return positionWiseBinaryOp(a, b, (o1, o2) => o1 - o2, resultShape);
}

export function multiply(
  a: CPUTensor,
  b: CPUTensor,
  resultShape: readonly number[]
) {
  return positionWiseBinaryOp(a, b, (o1, o2) => o1 * o2, resultShape);
}

export function divide(
  a: CPUTensor,
  b: CPUTensor,
  resultShape: readonly number[]
) {
  return positionWiseBinaryOp(a, b, (o1, o2) => o1 / o2, resultShape);
}

export function power(
  a: CPUTensor,
  b: CPUTensor,
  resultShape: readonly number[]
) {
  return positionWiseBinaryOp(a, b, (o1, o2) => Math.pow(o1, o2), resultShape);
}

export function clipBackward(
  value: CPUTensor,
  grad: CPUTensor,
  resultShape: readonly number[],
  min?: number,
  max?: number
) {
  return positionWiseBinaryOp(
    value,
    grad,
    (v, g) => {
      if (min !== undefined && v < min) {
        return 0;
      }
      if (max !== undefined && v > max) {
        return 0;
      }
      return g;
    },
    resultShape
  );
}
