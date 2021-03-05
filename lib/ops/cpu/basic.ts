import {CPUTensor} from '../../tensor/cpu/tensor';
import {DType} from '../../types';
import {checkEquivShapes, incrementIndex} from '../../util/shape';

// eslint-disable-next-line no-unused-vars
type UnaryOperator = (o: number) => number;
// eslint-disable-next-line no-unused-vars
type BinaryOperator = (o1: number, o2: number) => number;

export function positionWiseUnaryOp<DTpe extends DType>(
  a: CPUTensor<DTpe>,
  op: UnaryOperator
) {
  const result = new CPUTensor(a.shape, undefined, a.dtype);

  for (let i = 0; i < result.size; i += 1) {
    result.set(i, op(a.get(i)));
  }

  return result;
}

export function positionWiseBinaryOp<DTpe extends DType>(
  a: CPUTensor<DTpe>,
  b: CPUTensor<DTpe>,
  op: BinaryOperator,
  resultShape: readonly number[]
) {
  if (!checkEquivShapes(a.shape, b.shape)) {
    throw new Error(
      'The shapes of the two tensors should be the same for a binary operation'
    );
  }

  const result = new CPUTensor(resultShape, undefined, a.dtype);

  const index = new Array(resultShape.length).fill(0);

  for (let i = 0; i < result.size; i += 1) {
    result.set(index, op(a.get(index), b.get(index)));

    incrementIndex(index, resultShape);
  }

  return result;
}

export function exp<DTpe extends DType>(a: CPUTensor<DTpe>) {
  return positionWiseUnaryOp(a, o1 => Math.exp(o1));
}

export function log<DTpe extends DType>(a: CPUTensor<DTpe>) {
  return positionWiseUnaryOp(a, o1 => Math.log(o1));
}

export function sqrt<DTpe extends DType>(a: CPUTensor<DTpe>) {
  return positionWiseUnaryOp(a, o1 => Math.sqrt(o1));
}

export function abs<DTpe extends DType>(a: CPUTensor<DTpe>) {
  return positionWiseUnaryOp(a, o1 => Math.abs(o1));
}

export function sin<DTpe extends DType>(a: CPUTensor<DTpe>) {
  return positionWiseUnaryOp(a, o1 => Math.sin(o1));
}

export function cos<DTpe extends DType>(a: CPUTensor<DTpe>) {
  return positionWiseUnaryOp(a, o1 => Math.cos(o1));
}

export function tan<DTpe extends DType>(a: CPUTensor<DTpe>) {
  return positionWiseUnaryOp(a, o1 => Math.tan(o1));
}

export function asin<DTpe extends DType>(a: CPUTensor<DTpe>) {
  return positionWiseUnaryOp(a, o1 => Math.asin(o1));
}

export function acos<DTpe extends DType>(a: CPUTensor<DTpe>) {
  return positionWiseUnaryOp(a, o1 => Math.acos(o1));
}

export function atan<DTpe extends DType>(a: CPUTensor<DTpe>) {
  return positionWiseUnaryOp(a, o1 => Math.atan(o1));
}

export function sinh<DTpe extends DType>(a: CPUTensor<DTpe>) {
  return positionWiseUnaryOp(a, o1 => Math.sinh(o1));
}

export function cosh<DTpe extends DType>(a: CPUTensor<DTpe>) {
  return positionWiseUnaryOp(a, o1 => Math.cosh(o1));
}

export function tanh<DTpe extends DType>(a: CPUTensor<DTpe>) {
  return positionWiseUnaryOp(a, o1 => Math.tanh(o1));
}

export function asinh<DTpe extends DType>(a: CPUTensor<DTpe>) {
  return positionWiseUnaryOp(a, o1 => Math.asinh(o1));
}

export function acosh<DTpe extends DType>(a: CPUTensor<DTpe>) {
  return positionWiseUnaryOp(a, o1 => Math.acosh(o1));
}

export function atanh<DTpe extends DType>(a: CPUTensor<DTpe>) {
  return positionWiseUnaryOp(a, o1 => Math.atanh(o1));
}

export function floor<DTpe extends DType>(a: CPUTensor<DTpe>) {
  return positionWiseUnaryOp(a, o1 => Math.floor(o1));
}

export function ceil<DTpe extends DType>(a: CPUTensor<DTpe>) {
  return positionWiseUnaryOp(a, o1 => Math.ceil(o1));
}

export function round<DTpe extends DType>(a: CPUTensor<DTpe>) {
  return positionWiseUnaryOp(a, o1 => Math.round(o1));
}

export function sign<DTpe extends DType>(a: CPUTensor<DTpe>) {
  return positionWiseUnaryOp(a, o1 => (o1 < 0 ? -1 : o1 === 0 ? 0 : 1));
}

export function negate<DTpe extends DType>(a: CPUTensor<DTpe>) {
  return positionWiseUnaryOp(a, o1 => -o1);
}

export function powerScalar<DTpe extends DType>(
  a: CPUTensor<DTpe>,
  power: number,
  factor: number
) {
  return positionWiseUnaryOp(a, o1 => Math.pow(o1, power) * factor);
}

export function addMultiplyScalar<DTpe extends DType>(
  a: CPUTensor<DTpe>,
  factor: number,
  add: number
) {
  return positionWiseUnaryOp(a, o1 => o1 * factor + add);
}

export function sigmoid<DTpe extends DType>(a: CPUTensor<DTpe>) {
  return positionWiseUnaryOp(a, o1 => 1 / (1 + Math.exp(-o1)));
}

export function hardSigmoid<DTpe extends DType>(
  a: CPUTensor<DTpe>,
  alpha: number,
  beta: number
) {
  return positionWiseUnaryOp(a, o1 =>
    Math.max(0, Math.min(1, alpha * o1 + beta))
  );
}

export function clip<DTpe extends DType>(
  a: CPUTensor<DTpe>,
  min?: number,
  max?: number
) {
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

export function add<DTpe extends DType>(
  a: CPUTensor<DTpe>,
  b: CPUTensor<DTpe>,
  resultShape: readonly number[],
  alpha: number,
  beta: number
) {
  return positionWiseBinaryOp(
    a,
    b,
    (o1, o2) => o1 * alpha + o2 * beta,
    resultShape
  );
}

export function subtract<DTpe extends DType>(
  a: CPUTensor<DTpe>,
  b: CPUTensor<DTpe>,
  resultShape: readonly number[],
  alpha: number,
  beta: number
) {
  return positionWiseBinaryOp(
    a,
    b,
    (o1, o2) => o1 * alpha - o2 * beta,
    resultShape
  );
}

export function multiply<DTpe extends DType>(
  a: CPUTensor<DTpe>,
  b: CPUTensor<DTpe>,
  resultShape: readonly number[],
  alpha: number
) {
  return positionWiseBinaryOp(a, b, (o1, o2) => o1 * o2 * alpha, resultShape);
}

export function divide<DTpe extends DType>(
  a: CPUTensor<DTpe>,
  b: CPUTensor<DTpe>,
  resultShape: readonly number[],
  alpha: number
) {
  return positionWiseBinaryOp(a, b, (o1, o2) => (o1 / o2) * alpha, resultShape);
}

export function power<DTpe extends DType>(
  a: CPUTensor<DTpe>,
  b: CPUTensor<DTpe>,
  resultShape: readonly number[]
) {
  return positionWiseBinaryOp(a, b, (o1, o2) => Math.pow(o1, o2), resultShape);
}

export function clipBackward<DTpe extends DType>(
  value: CPUTensor<DTpe>,
  grad: CPUTensor<DTpe>,
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
