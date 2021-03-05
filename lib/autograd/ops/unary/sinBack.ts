import {DType, Tensor} from '../../../library';
import {BackwardOp, VariableI} from '../../types';

export class SinBack<DTpe extends DType> implements BackwardOp<DTpe> {
  constructor(public input: VariableI<DTpe>) {}

  backward(grad: Tensor<DTpe>): void {
    const cos = this.input.value.cos();
    const gradAbs = grad.multiply(cos);
    cos.delete();
    const needed = this.input.backward(gradAbs);
    if (!needed) {
      gradAbs.delete();
    }
  }

  delete(): void {
    if (!this.input.isLeaf()) {
      this.input.delete();
    }
  }
}

export class ASinBack<DTpe extends DType> implements BackwardOp<DTpe> {
  constructor(public input: VariableI<DTpe>) {}

  backward(grad: Tensor<DTpe>): void {
    const squared = this.input.value.multiply(this.input.value);
    const oneMinus = squared.addMultiplyScalar(-1, 1);
    squared.delete();
    const sqrt = oneMinus.sqrt();
    oneMinus.delete();
    const gradASin = grad.divide(sqrt);
    sqrt.delete();
    const needed = this.input.backward(gradASin);
    if (!needed) {
      gradASin.delete();
    }
  }

  delete(): void {
    if (!this.input.isLeaf()) {
      this.input.delete();
    }
  }
}

export class SinHBack<DTpe extends DType> implements BackwardOp<DTpe> {
  constructor(public input: VariableI<DTpe>) {}

  backward(grad: Tensor<DTpe>): void {
    const cosh = this.input.value.cosh();
    const gradCosH = grad.multiply(cosh);
    cosh.delete();
    const needed = this.input.backward(gradCosH);
    if (!needed) {
      gradCosH.delete();
    }
  }

  delete(): void {
    if (!this.input.isLeaf()) {
      this.input.delete();
    }
  }
}

export class ASinHBack<DTpe extends DType> implements BackwardOp<DTpe> {
  constructor(public input: VariableI<DTpe>) {}

  backward(grad: Tensor<DTpe>): void {
    const squared = this.input.value.multiply(this.input.value);
    const onePlus = squared.addMultiplyScalar(1, 1);
    squared.delete();
    const sqrt = onePlus.sqrt();
    onePlus.delete();
    const gradASinH = grad.divide(sqrt);
    sqrt.delete();
    const needed = this.input.backward(gradASinH);
    if (!needed) {
      gradASinH.delete();
    }
  }

  delete(): void {
    if (!this.input.isLeaf()) {
      this.input.delete();
    }
  }
}
