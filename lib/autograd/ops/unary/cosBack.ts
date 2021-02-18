import {Tensor} from '../../../library';
import {BackwardOp, VariableI} from '../../types';

export class CosBack implements BackwardOp {
  constructor(public input: VariableI) {}

  backward(grad: Tensor): void {
    const sin = this.input.value.sin();
    const gradAbs = grad.multiply(sin, -1);
    sin.delete();
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

export class ACosBack implements BackwardOp {
  constructor(public input: VariableI) {}

  backward(grad: Tensor): void {
    const squared = this.input.value.multiply(this.input.value);
    const oneMinus = squared.addMultiplyScalar(-1, 1);
    squared.delete();
    const sqrt = oneMinus.sqrt();
    oneMinus.delete();
    const gradACos = grad.divide(sqrt, -1);
    sqrt.delete();
    const needed = this.input.backward(gradACos);
    if (!needed) {
      gradACos.delete();
    }
  }

  delete(): void {
    if (!this.input.isLeaf()) {
      this.input.delete();
    }
  }
}

export class CosHBack implements BackwardOp {
  constructor(public input: VariableI) {}

  backward(grad: Tensor): void {
    const sinh = this.input.value.sinh();
    const gradCosH = grad.multiply(sinh);
    sinh.delete();
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

export class ACosHBack implements BackwardOp {
  constructor(public input: VariableI) {}

  backward(grad: Tensor): void {
    const squared = this.input.value.multiply(this.input.value);
    const onePlus = squared.addMultiplyScalar(1, -1);
    squared.delete();
    const sqrt = onePlus.sqrt();
    onePlus.delete();
    const gradACosH = grad.divide(sqrt);
    sqrt.delete();
    const needed = this.input.backward(gradACosH);
    if (!needed) {
      gradACosH.delete();
    }
  }

  delete(): void {
    if (!this.input.isLeaf()) {
      this.input.delete();
    }
  }
}
