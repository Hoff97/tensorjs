import {Tensor} from '../../../library';
import {BackwardOp, VariableI} from '../../types';

export class TanBack implements BackwardOp {
  constructor(public input: VariableI) {}

  backward(grad: Tensor): void {
    const cos = this.input.value.cos();
    const cos2 = cos.multiply(cos);
    cos.delete();
    const gradTan = grad.divide(cos2);
    cos2.delete();
    const needed = this.input.backward(gradTan);
    if (!needed) {
      gradTan.delete();
    }
  }

  delete(): void {
    if (!this.input.isLeaf()) {
      this.input.delete();
    }
  }
}

export class ATanBack implements BackwardOp {
  constructor(public input: VariableI) {}

  backward(grad: Tensor): void {
    const squared = this.input.value.multiply(this.input.value);
    const onePlus = squared.addMultiplyScalar(1, 1);
    squared.delete();
    const gradATan = grad.divide(onePlus);
    onePlus.delete();
    const needed = this.input.backward(gradATan);
    if (!needed) {
      gradATan.delete();
    }
  }

  delete(): void {
    if (!this.input.isLeaf()) {
      this.input.delete();
    }
  }
}

export class TanHBack implements BackwardOp {
  constructor(public input: VariableI, public tanH: Tensor) {}

  backward(grad: Tensor): void {
    const squared = this.tanH.multiply(this.tanH);
    const onePlus = squared.addMultiplyScalar(-1, 1);
    squared.delete();
    const gradTanH = grad.multiply(onePlus);
    onePlus.delete();
    const needed = this.input.backward(gradTanH);
    if (!needed) {
      gradTanH.delete();
    }
  }

  delete(): void {
    if (!this.input.isLeaf()) {
      this.input.delete();
    }
  }
}

export class ATanHBack implements BackwardOp {
  constructor(public input: VariableI) {}

  backward(grad: Tensor): void {
    const squared = this.input.value.multiply(this.input.value);
    const onePlus = squared.addMultiplyScalar(-1, 1);
    squared.delete();
    const gradATanH = grad.divide(onePlus);
    onePlus.delete();
    const needed = this.input.backward(gradATanH);
    if (!needed) {
      gradATanH.delete();
    }
  }

  delete(): void {
    if (!this.input.isLeaf()) {
      this.input.delete();
    }
  }
}
