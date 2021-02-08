import {Tensor} from '../../../library';
import {BackwardOp, VariableI} from '../../types';

export class ConvBack implements BackwardOp {
  constructor(
    public x: VariableI,
    public w: VariableI,
    public strides: number[],
    public padding: number[],
    public dilations: number[],
    public group: number,
    public b?: VariableI
  ) {}

  backward(grad: Tensor): void {
    if (!this.w.noGrad) {
      const gradW = this.x.value.conv(
        grad,
        undefined,
        this.strides,
        this.group,
        this.padding,
        this.dilations
      );
      const needed = this.w.backward(gradW);
      if (!needed) {
        gradW.delete();
      }
    }

    if (this.b !== undefined && !this.b.noGrad) {
      const biasSum = [0];
      for (let i = 0; i < this.dilations.length; i++) {
        biasSum.push(i + 2);
      }

      const gradB = grad.sum(biasSum);
      const needed = this.b.backward(gradB);
      if (!needed) {
        gradB.delete();
      }
    }

    if (!this.x.noGrad) {
      const wShape = this.w.getShape();

      let xPads = [];
      for (let i = 0; i < this.dilations.length; i++) {
        xPads.push(wShape[i + 2] - this.padding[i] + this.dilations[i] - 2);
      }
      xPads = [...xPads, ...xPads];

      const gradX = grad.convTranspose(
        this.w.value,
        this.dilations,
        this.group,
        xPads,
        this.strides
      );
      const needed = this.x.backward(gradX);
      if (!needed) {
        gradX.delete();
      }
    }
  }

  delete(): void {
    if (!this.x.isLeaf()) {
      this.x.delete();
    }

    if (!this.w.isLeaf()) {
      this.w.delete();
    }

    if (this.b !== undefined && !this.b.isLeaf()) {
      this.b.delete();
    }
  }
}
