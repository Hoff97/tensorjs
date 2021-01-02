import CPUTensor from '../../tensor/cpu/tensor';
import { compareShapes } from '../../util/shape';

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

export function positionWiseBinaryOp(a: CPUTensor, b: CPUTensor, op: BinaryOperator) {
  if (!compareShapes(a.shape, b.shape)) {
    throw new Error('The shapes of the two tensors should be the same for a binary operation');
  }

  const result = new CPUTensor(a.shape);

  for (let i = 0; i < result.size; i += 1) {
    result.set(i, op(a.get(i), b.get(i)));
  }

  return result;
}

export function exp(a: CPUTensor) {
  return positionWiseUnaryOp(a, (o1) => Math.exp(o1));
}

export function log(a: CPUTensor) {
  return positionWiseUnaryOp(a, (o1) => Math.log(o1));
}

export function sqrt(a: CPUTensor) {
  return positionWiseUnaryOp(a, (o1) => Math.sqrt(o1));
}

export function add(a: CPUTensor, b: CPUTensor) {
  return positionWiseBinaryOp(a, b, (o1, o2) => o1 + o2);
}

export function subtract(a: CPUTensor, b: CPUTensor) {
  return positionWiseBinaryOp(a, b, (o1, o2) => o1 - o2);
}

export function multiply(a: CPUTensor, b: CPUTensor) {
  return positionWiseBinaryOp(a, b, (o1, o2) => o1 * o2);
}

export function divide(a: CPUTensor, b: CPUTensor) {
  return positionWiseBinaryOp(a, b, (o1, o2) => o1 / o2);
}

